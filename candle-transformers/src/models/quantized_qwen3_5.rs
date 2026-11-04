//! Qwen3.5 and Qwen3.8 dense hybrid models from llama.cpp GGUF files (architecture `qwen35`),
//! including PrismML's Ternary Bonsai models.
//!
//! The layers are those of [`super::qwen3_5`], with the weights laid out as llama.cpp's converter
//! writes them: the norm weights hold `1 + w`, `ssm_a` holds `-exp(A_log)`, and the value heads of
//! the Gated DeltaNet layers are in tiled order, value head `r * n_k + g` reading key head `g`.
//! The MTP layers are the `nextn` blocks after the trunk, and can carry a trimmed draft head that
//! only proposes its most frequent tokens.
//!
//! PrismML's llama.cpp fork adds the ternary weight types of [`candle::quantized::ternary`] and
//! folded Hadamard rotations (`prism.hadamard.*` metadata): a folded weight `W D H` reads the
//! input rotated by the signs `D` and the normalized block Walsh-Hadamard matrix `H`.
//!
//! - 💻 [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/prism):
//!   `src/models/qwen35.cpp`, `src/llama-model.cpp`, `conversion/qwen.py`
//! - 🤗 [Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)

use super::qwen3_5::{
    chunk_gated_delta_rule, generate_mtp, recurrent_gated_delta_rule, softplus, MtpModel, MtpStats,
    RotaryEmbedding,
};
use super::qwen3_5_vl::{prompt_embeds, Images, VisionModel};
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QMatMul};
use candle::{DType, Device, Module, Result, Tensor, D};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek};
use std::sync::Arc;

/// How the Gated DeltaNet layers normalize their queries and keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Norm {
    /// `x / max(|x|, eps)`, ggml's `l2_norm`: the PrismML fork keeps it for the models it packs.
    Max,
    /// `x / sqrt(|x|^2 + eps)`, as transformers and upstream llama.cpp.
    Sqrt,
}

/// The hyper-parameters of a `qwen35` GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// The trunk layers, without the MTP layers.
    pub num_hidden_layers: usize,
    pub num_mtp_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// The rotated features of each head.
    pub rope_dim: usize,
    pub max_position_embeddings: usize,
    pub conv_kernel: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    /// Whether each trunk layer is a Gated DeltaNet layer.
    pub recurrent: Vec<bool>,
    /// [`L2Norm::Max`] from [`Config::from_gguf`], for the Ternary Bonsai models.
    pub l2_norm: L2Norm,
    /// The temporal, height and width frequency sections of multimodal RoPE.
    pub mrope_section: Option<Vec<usize>>,
}

fn int(v: &gguf_file::Value) -> Result<i64> {
    use gguf_file::Value as V;
    Ok(match v {
        V::U8(v) => *v as i64,
        V::I8(v) => *v as i64,
        V::U16(v) => *v as i64,
        V::I16(v) => *v as i64,
        V::U32(v) => *v as i64,
        V::I32(v) => *v as i64,
        V::U64(v) => *v as i64,
        V::I64(v) => *v,
        v => candle::bail!("not an integer {v:?}"),
    })
}

fn float(v: &gguf_file::Value) -> Result<f64> {
    use gguf_file::Value as V;
    match v {
        V::F32(v) => Ok(*v as f64),
        V::F64(v) => Ok(*v),
        v => Ok(int(v)? as f64),
    }
}

impl Config {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let md = &ct.metadata;
        let arch = md.get("general.architecture").map(|v| v.to_string());
        if let Some(Ok(arch)) = arch {
            if arch != "qwen35" {
                candle::bail!("expected a qwen35 GGUF, got {arch}")
            }
        }
        let get = |k: &str| match md.get(&format!("qwen35.{k}")) {
            Some(v) => Ok(v),
            None => candle::bail!("cannot find qwen35.{k} in the metadata"),
        };
        let usize = |k: &str| -> Result<usize> { Ok(int(get(k)?)? as usize) };
        let block_count = usize("block_count")?;
        let num_mtp_layers = md
            .get("qwen35.nextn_predict_layers")
            .map(int)
            .transpose()?
            .unwrap_or(0) as usize;
        let num_hidden_layers = block_count - num_mtp_layers;
        let recurrent = match md.get("qwen35.attention.recurrent_layers") {
            Some(v) => v
                .to_vec()?
                .iter()
                .take(num_hidden_layers)
                .map(|v| v.to_bool())
                .collect::<Result<Vec<_>>>()?,
            None => {
                let interval = match md.get("qwen35.full_attention_interval") {
                    Some(v) => int(v)? as usize,
                    None => 4,
                };
                (0..num_hidden_layers)
                    .map(|i| (i + 1) % interval != 0)
                    .collect()
            }
        };
        if recurrent.len() != num_hidden_layers {
            candle::bail!(
                "{} recurrent flags for {num_hidden_layers} layers",
                recurrent.len()
            )
        }
        let vocab_size = match ct.tensor_infos.get("token_embd.weight") {
            Some(info) => info.shape.dims()[0],
            None => candle::bail!("cannot find token_embd.weight"),
        };
        let num_v_heads = usize("ssm.time_step_rank")?;
        Ok(Self {
            vocab_size,
            hidden_size: usize("embedding_length")?,
            intermediate_size: usize("feed_forward_length")?,
            num_hidden_layers,
            num_mtp_layers,
            num_attention_heads: usize("attention.head_count")?,
            num_key_value_heads: usize("attention.head_count_kv")?,
            head_dim: usize("attention.key_length")?,
            rms_norm_eps: float(get("attention.layer_norm_rms_epsilon")?)?,
            rope_theta: float(get("rope.freq_base")?)?,
            rope_dim: usize("rope.dimension_count")?,
            max_position_embeddings: usize("context_length")?,
            conv_kernel: usize("ssm.conv_kernel")?,
            num_k_heads: usize("ssm.group_count")?,
            num_v_heads,
            head_k_dim: usize("ssm.state_size")?,
            head_v_dim: usize("ssm.inner_size")? / num_v_heads,
            recurrent,
            l2_norm: L2Norm::Max,
            mrope_section: match md.get("qwen35.rope.dimension_sections") {
                // llama.cpp stores four sections, the last one unused here.
                Some(v) => Some(
                    v.to_vec()?
                        .iter()
                        .take(3)
                        .map(|v| Ok(int(v)? as usize))
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            },
        })
    }
}

/// The input transform of a Hadamard-folded weight: the head permutation of a grouped GDN
/// output projection, the signs, then the normalized block Walsh-Hadamard rotation.
#[derive(Debug, Clone)]
struct HadamardInput {
    /// `(block, block)`, symmetric.
    h: Tensor,
    signs: Option<Tensor>,
    /// `(head_dim, n_k, rep)`: the input holds its value heads in tiled order `r * n_k + g` and
    /// the fold expects the grouped order `g * rep + r`.
    gdn_perm: Option<(usize, usize, usize)>,
}

impl HadamardInput {
    fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let n = *dims.last().unwrap();
        let m = xs.elem_count() / n;
        let mut x = xs.reshape((m, n))?;
        if let Some((hd, nk, rep)) = self.gdn_perm {
            x = x
                .reshape((m, rep, nk, hd))?
                .transpose(1, 2)?
                .reshape((m, n))?;
        }
        if let Some(signs) = &self.signs {
            x = x.broadcast_mul(signs)?
        }
        x.reshape((m, n / self.h.dim(0)?, self.h.dim(0)?))?
            .broadcast_matmul(&self.h)?
            .reshape(dims)
    }

    /// The inverse, on rows looked up in a rotated table: the rotation, then the signs.
    fn invert(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let n = *dims.last().unwrap();
        let b = self.h.dim(0)?;
        let x = xs
            .reshape((xs.elem_count() / b, b))?
            .matmul(&self.h)?
            .reshape(dims)?;
        match &self.signs {
            Some(signs) => x.broadcast_mul(&signs.reshape(n)?),
            None => Ok(x),
        }
    }
}

/// The `prism.hadamard` metadata.
#[derive(Debug, Clone)]
struct Hadamard {
    h: Tensor,
    block_size: usize,
    /// Sign vectors by width.
    signs: HashMap<usize, Tensor>,
    /// Folded weights, reading rotated inputs.
    folded: HashSet<String>,
    /// Tables whose looked-up rows are rotated.
    inverse: HashSet<String>,
    gdn_v_grouped: bool,
}

impl Hadamard {
    fn from_gguf(ct: &gguf_file::Content, device: &Device) -> Result<Option<Self>> {
        let md = &ct.metadata;
        let key = |k: &str| md.get(&format!("prism.hadamard.{k}"));
        let Some(version) = key("version") else {
            return Ok(None);
        };
        let version = int(version)?;
        let tied_output = key("tied_output").map(|v| v.to_bool()).transpose()?;
        let tied_output = tied_output.unwrap_or(false);
        if version != 1 && version != 2 {
            candle::bail!("unsupported prism.hadamard.version {version}")
        }
        if (version == 2) != tied_output {
            candle::bail!("prism.hadamard version 2 requires tied_output, version 1 forbids it")
        }
        let string = |k: &str| -> Result<String> {
            match key(k) {
                Some(v) => Ok(v.to_string()?.clone()),
                None => candle::bail!("cannot find prism.hadamard.{k}"),
            }
        };
        let strings = |k: &str| -> Result<Vec<String>> {
            match key(k) {
                Some(v) => v
                    .to_vec()?
                    .iter()
                    .map(|v| Ok(v.to_string()?.clone()))
                    .collect(),
                None => Ok(vec![]),
            }
        };
        let ints = |k: &str| -> Result<Vec<i64>> {
            match key(k) {
                Some(v) => v.to_vec()?.iter().map(int).collect(),
                None => Ok(vec![]),
            }
        };
        let block_size = match key("block_size") {
            Some(v) => int(v)? as usize,
            None => candle::bail!("cannot find prism.hadamard.block_size"),
        };
        if !block_size.is_power_of_two() {
            candle::bail!("invalid prism.hadamard.block_size {block_size}")
        }
        let transform = string("transform")?;
        if transform != "normalized-sylvester-walsh-hadamard" {
            candle::bail!("unsupported prism.hadamard.transform {transform}")
        }
        let axis = string("axis")?;
        if axis != "input-last-dimension" {
            candle::bail!("unsupported prism.hadamard.axis {axis}")
        }
        let mut signs = HashMap::new();
        match string("sign_mode")?.as_str() {
            "identity" => {}
            "explicit" => {
                let (widths, values) = (ints("sign_widths")?, ints("sign_values")?);
                if widths.is_empty() {
                    candle::bail!("prism.hadamard.sign_mode is explicit without sign widths")
                }
                let mut offset = 0;
                for w in widths {
                    let w = w as usize;
                    if w == 0 || !w.is_multiple_of(block_size) || offset + w > values.len() {
                        candle::bail!("invalid prism.hadamard sign width {w}")
                    }
                    let v = &values[offset..offset + w];
                    if v.iter().any(|&s| s != 1 && s != -1) {
                        candle::bail!("prism.hadamard sign values must be +-1")
                    }
                    let v: Vec<f32> = v.iter().map(|&s| s as f32).collect();
                    if signs.insert(w, Tensor::new(v, device)?).is_some() {
                        candle::bail!("duplicate prism.hadamard sign width {w}")
                    }
                    offset += w
                }
                if offset != values.len() {
                    candle::bail!("prism.hadamard.sign_values length mismatch")
                }
            }
            mode => candle::bail!("unsupported prism.hadamard.sign_mode {mode}"),
        }
        let mut folded: HashSet<String> = strings("weight_names")?.into_iter().collect();
        if folded.is_empty() {
            candle::bail!("prism.hadamard.weight_names is empty")
        }
        let inverse: HashSet<String> = strings("inverse_weight_names")?.into_iter().collect();
        if inverse.iter().any(|n| n != "token_embd.weight") {
            candle::bail!("prism.hadamard only inverts the token embedding lookup")
        }
        if tied_output {
            if !inverse.contains("token_embd.weight") {
                candle::bail!("prism.hadamard.tied_output requires a rotated token embedding")
            }
            if ct.tensor_infos.contains_key("output.weight") {
                candle::bail!("prism.hadamard.tied_output requires output.weight to be absent")
            }
            folded.insert("token_embd.weight".to_string());
        }
        let gdn_v_grouped = key("gdn_v_grouped").map(|v| v.to_bool()).transpose()?;
        // H[i, j] = (-1)^popcount(i & j) / sqrt(block_size).
        let scale = 1. / (block_size as f32).sqrt();
        let h: Vec<f32> = (0..block_size * block_size)
            .map(|k| {
                let (i, j) = (k / block_size, k % block_size);
                if (i & j).count_ones() % 2 == 1 {
                    -scale
                } else {
                    scale
                }
            })
            .collect();
        Ok(Some(Self {
            h: Tensor::from_vec(h, (block_size, block_size), device)?,
            block_size,
            signs,
            folded,
            inverse,
            gdn_v_grouped: gdn_v_grouped.unwrap_or(false),
        }))
    }

    fn input(&self, width: usize, name: &str) -> Result<HadamardInput> {
        if !width.is_multiple_of(self.block_size) {
            candle::bail!(
                "prism.hadamard block size {} does not divide the width {width} of {name}",
                self.block_size
            )
        }
        let signs = if self.signs.is_empty() {
            None
        } else {
            match self.signs.get(&width) {
                Some(s) => Some(s.clone()),
                None => candle::bail!("prism.hadamard has no sign vector of width {width}"),
            }
        };
        Ok(HadamardInput {
            h: self.h.clone(),
            signs,
            gdn_perm: None,
        })
    }
}

/// A quantized matmul, with the input transform of a Hadamard-folded weight and a per-tensor
/// output scale when the file has them.
#[derive(Debug, Clone)]
struct QLinear {
    w: QMatMul,
    input: Option<HadamardInput>,
    scale: Option<Tensor>,
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = match &self.input {
            Some(t) => self.w.forward(&t.apply(xs)?)?,
            None => self.w.forward(xs)?,
        };
        match &self.scale {
            Some(s) => ys.broadcast_mul(s),
            None => Ok(ys),
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&xs.contiguous()?, &self.weight, self.eps)
    }
}

struct Weights<'a, R: Read + Seek> {
    ct: &'a gguf_file::Content,
    reader: &'a mut R,
    device: Device,
    hadamard: Option<Hadamard>,
    eps: f32,
}

impl<R: Read + Seek> Weights<'_, R> {
    fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    fn f32(&mut self, name: &str) -> Result<Tensor> {
        self.ct
            .tensor(self.reader, name, &self.device)?
            .dequantize(&self.device)
    }

    fn norm(&mut self, name: &str) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.f32(name)?,
            eps: self.eps,
        })
    }

    /// The matmul of `{base}.weight`, its `{base}.scale` and its Hadamard input transform.
    fn linear(&mut self, base: &str) -> Result<QLinear> {
        let name = format!("{base}.weight");
        let w = self.ct.tensor(self.reader, &name, &self.device)?;
        let width = w.shape().dims()[1];
        let input = match &self.hadamard {
            Some(h) if h.folded.contains(&name) => Some(h.input(width, &name)?),
            _ => None,
        };
        let scale_name = format!("{base}.scale");
        let scale = if self.has(&scale_name) {
            Some(self.f32(&scale_name)?.flatten_all()?)
        } else {
            None
        };
        Ok(QLinear {
            w: QMatMul::from_qtensor(w)?,
            input,
            scale,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        (gate * self.up.forward(xs)?)?.apply(&self.down)
    }
}

#[derive(Debug, Clone)]
enum Qkv {
    Fused(QLinear),
    Split(QLinear, QLinear, QLinear),
}

/// Full attention with QK-norm, partial RoPE and a sigmoid output gate; the query projection
/// gives each head's query then its gate.
#[derive(Debug, Clone)]
struct Attention {
    qkv: Qkv,
    o: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new<R: Read + Seek>(
        w: &mut Weights<R>,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        p: &str,
    ) -> Result<Self> {
        let qkv = if w.has(&format!("{p}.attn_qkv.weight")) {
            Qkv::Fused(w.linear(&format!("{p}.attn_qkv"))?)
        } else {
            Qkv::Split(
                w.linear(&format!("{p}.attn_q"))?,
                w.linear(&format!("{p}.attn_k"))?,
                w.linear(&format!("{p}.attn_v"))?,
            )
        };
        Ok(Self {
            qkv,
            o: w.linear(&format!("{p}.attn_output"))?,
            q_norm: w.norm(&format!("{p}.attn_q_norm.weight"))?,
            k_norm: w.norm(&format!("{p}.attn_k_norm.weight"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, rope: &Rope) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let (q, k, v) = match &self.qkv {
            Qkv::Fused(qkv) => {
                let qkv = qkv.forward(xs)?;
                (
                    qkv.narrow(2, 0, 2 * nh * hd)?,
                    qkv.narrow(2, 2 * nh * hd, nkv * hd)?,
                    qkv.narrow(2, 2 * nh * hd + nkv * hd, nkv * hd)?,
                )
            }
            Qkv::Split(q, k, v) => (q.forward(xs)?, k.forward(xs)?, v.forward(xs)?),
        };
        let q = q.reshape((b, l, nh, 2 * hd))?;
        let gate = q.narrow(3, hd, hd)?.reshape((b, l, nh * hd))?;
        let q = self.q_norm.forward(&q.narrow(3, 0, hd)?)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k.reshape((b, l, nkv, hd))?)?;
        let k = k.transpose(1, 2)?;
        let v = v.reshape((b, l, nkv, hd))?.transpose(1, 2)?.contiguous()?;
        let q = self
            .rotary
            .apply_cos_sin(&q.contiguous()?, &rope.0, &rope.1)?;
        let k = self
            .rotary
            .apply_cos_sin(&k.contiguous()?, &rope.0, &rope.1)?;
        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let k = repeat_kv(k, nh / nkv)?.contiguous()?;
        let v = repeat_kv(v, nh / nkv)?.contiguous()?;
        let scores = (q.contiguous()?.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ys = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, nh * hd))?;
        self.o
            .forward(&(ys * candle_nn::ops::sigmoid(&gate.contiguous()?)?)?)
    }
}

/// The norm of [`L2Norm`] over the last dimension.
fn l2_norm(xs: &Tensor, kind: L2Norm, eps: f64) -> Result<Tensor> {
    let sq = xs.sqr()?.sum_keepdim(D::Minus1)?;
    let norm = match kind {
        L2Norm::Max => sq.sqrt()?.maximum(eps)?,
        L2Norm::Sqrt => (sq + eps)?.sqrt()?,
    };
    xs.broadcast_div(&norm)
}

/// The Gated DeltaNet layer, with its value heads in tiled order.
#[derive(Debug, Clone)]
struct GatedDeltaNet {
    qkv: QLinear,
    z: QLinear,
    beta: QLinear,
    alpha: QLinear,
    out: QLinear,
    /// `(conv_dim, kernel)`.
    conv_weight: Tensor,
    /// `-exp(A_log)`.
    a: Tensor,
    dt_bias: Tensor,
    norm: RmsNorm,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f64,
    l2_norm: L2Norm,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNet {
    fn new<R: Read + Seek>(w: &mut Weights<R>, cfg: &Config, p: &str) -> Result<Self> {
        let mut out = w.linear(&format!("{p}.ssm_out"))?;
        let grouped = w.hadamard.as_ref().is_some_and(|h| h.gdn_v_grouped);
        if let (Some(input), true) = (&mut out.input, grouped) {
            let (nk, nv) = (cfg.num_k_heads, cfg.num_v_heads);
            input.gdn_perm = Some((cfg.head_v_dim, nk, nv / nk));
        }
        Ok(Self {
            qkv: w.linear(&format!("{p}.attn_qkv"))?,
            z: w.linear(&format!("{p}.attn_gate"))?,
            beta: w.linear(&format!("{p}.ssm_beta"))?,
            alpha: w.linear(&format!("{p}.ssm_alpha"))?,
            out,
            conv_weight: w.f32(&format!("{p}.ssm_conv1d.weight"))?,
            a: w.f32(&format!("{p}.ssm_a"))?,
            dt_bias: w.f32(&format!("{p}.ssm_dt.bias"))?,
            norm: w.norm(&format!("{p}.ssm_norm.weight"))?,
            num_k_heads: cfg.num_k_heads,
            num_v_heads: cfg.num_v_heads,
            head_k_dim: cfg.head_k_dim,
            head_v_dim: cfg.head_v_dim,
            eps: cfg.rms_norm_eps,
            l2_norm: cfg.l2_norm,
            conv_state: None,
            recurrent_state: None,
        })
    }

    /// The causal depthwise convolution followed by SiLU, continuing from the cached inputs.
    fn conv(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, conv_dim, l) = xs.dims3()?;
        let kernel = self.conv_weight.dim(1)?;
        let state = match &self.conv_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((b, conv_dim, kernel - 1), xs.dtype(), xs.device())?,
        };
        let full = Tensor::cat(&[&state, xs], 2)?;
        let mut ys = full
            .narrow(2, 0, l)?
            .broadcast_mul(&self.conv_weight.narrow(1, 0, 1)?)?;
        for j in 1..kernel {
            let w = self.conv_weight.narrow(1, j, 1)?;
            ys = (ys + full.narrow(2, j, l)?.broadcast_mul(&w)?)?;
        }
        self.conv_state = Some(full.narrow(2, l, kernel - 1)?.contiguous()?);
        candle_nn::ops::silu(&ys)
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let (nk, nv, dk, dv) = (
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        );
        let mixed = self.qkv.forward(xs)?.transpose(1, 2)?.contiguous()?;
        let mixed = self.conv(&mixed)?.transpose(1, 2)?;
        let z = self.z.forward(xs)?;
        let beta = candle_nn::ops::sigmoid(&self.beta.forward(xs)?)?;
        let alpha = self.alpha.forward(xs)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        // (batch, seq, heads, dim) -> (batch, heads, seq, dim). The value heads are tiled, so
        // the key heads repeat as a whole.
        let heads = |xs: Tensor, n: usize, d: usize, norm: bool| -> Result<Tensor> {
            let xs = xs.reshape((b, l, n, d))?;
            let xs = if norm {
                l2_norm(&xs, self.l2_norm, self.eps)?
            } else {
                xs
            };
            let xs = if n < nv {
                Tensor::cat(&vec![&xs; nv / n], 2)?
            } else {
                xs
            };
            xs.transpose(1, 2)?.contiguous()
        };
        let q = heads(mixed.narrow(2, 0, nk * dk)?, nk, dk, true)?;
        let k = heads(mixed.narrow(2, nk * dk, nk * dk)?, nk, dk, true)?;
        let v = heads(mixed.narrow(2, 2 * nk * dk, nv * dv)?, nv, dv, false)?;
        let q = (q * (1. / (dk as f64).sqrt()))?;
        let beta = beta.transpose(1, 2)?.contiguous()?;
        let g = g.transpose(1, 2)?.contiguous()?;
        let state = match &self.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((b, nv, dk, dv), DType::F32, xs.device())?,
        };
        let (out, state) = if l == 1 {
            recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, &state)?
        } else {
            chunk_gated_delta_rule(&q, &k, &v, &g, &beta, &state, 64)?
        };
        self.recurrent_state = Some(state);
        let out = out.transpose(1, 2)?.reshape((b, l, nv, dv))?;
        let z = candle_nn::ops::silu(&z.reshape((b, l, nv, dv))?)?;
        let out = (self.norm.forward(&out)? * z)?;
        self.out.forward(&out.reshape((b, l, nv * dv))?)
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    Linear(GatedDeltaNet),
    Full(Attention),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    mixer: TokenMixer,
    mlp: Mlp,
    attn_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl DecoderLayer {
    fn new<R: Read + Seek>(
        w: &mut Weights<R>,
        cfg: &Config,
        recurrent: bool,
        rotary: Arc<RotaryEmbedding>,
        p: &str,
    ) -> Result<Self> {
        let mixer = if recurrent {
            TokenMixer::Linear(GatedDeltaNet::new(w, cfg, p)?)
        } else {
            TokenMixer::Full(Attention::new(w, cfg, rotary, p)?)
        };
        Ok(Self {
            mixer,
            mlp: Mlp {
                gate: w.linear(&format!("{p}.ffn_gate"))?,
                up: w.linear(&format!("{p}.ffn_up"))?,
                down: w.linear(&format!("{p}.ffn_down"))?,
            },
            attn_norm: w.norm(&format!("{p}.attn_norm.weight"))?,
            post_norm: w.norm(&format!("{p}.post_attention_norm.weight"))?,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, rope: &Rope) -> Result<Tensor> {
        let h = self.attn_norm.forward(xs)?;
        let h = match &mut self.mixer {
            TokenMixer::Linear(m) => m.forward(&h)?,
            TokenMixer::Full(m) => m.forward(&h, mask, rope)?,
        };
        let xs = (xs + h)?;
        let h = self.post_norm.forward(&xs)?.apply(&self.mlp)?;
        xs + h
    }

    fn cache(&self) -> LayerCache {
        match &self.mixer {
            TokenMixer::Linear(m) => {
                LayerCache::Linear(m.conv_state.clone(), m.recurrent_state.clone())
            }
            TokenMixer::Full(m) => LayerCache::Full(m.kv_cache.clone()),
        }
    }

    fn set_cache(&mut self, cache: LayerCache) -> Result<()> {
        match (&mut self.mixer, cache) {
            (TokenMixer::Linear(m), LayerCache::Linear(conv, recurrent)) => {
                m.conv_state = conv;
                m.recurrent_state = recurrent;
            }
            (TokenMixer::Full(m), LayerCache::Full(kv)) => m.kv_cache = kv,
            _ => candle::bail!("cache of another layer type"),
        }
        Ok(())
    }
}

/// The `(seq, rope_dim / 2)` cos and sin of the positions of a forward pass.
type Rope = (Tensor, Tensor);

#[derive(Debug, Clone)]
enum LayerCache {
    Linear(Option<Tensor>, Option<Tensor>),
    Full(Option<(Tensor, Tensor)>),
}

/// A snapshot of every cache of a [`Model`], MTP layers included.
#[derive(Debug, Clone)]
pub struct Cache {
    layers: Vec<LayerCache>,
    mtp_layers: Vec<LayerCache>,
}

/// A table lookup, with the inverse Hadamard transform of a rotated table.
#[derive(Debug, Clone)]
struct Embedding {
    table: QMatMul,
    inverse: Option<HadamardInput>,
}

impl Embedding {
    fn new<R: Read + Seek>(w: &mut Weights<R>, name: &str) -> Result<Self> {
        let table = w.ct.tensor(w.reader, name, &w.device)?;
        let width = table.shape().dims()[1];
        let inverse = match &w.hadamard {
            Some(h) if h.inverse.contains(name) => Some(h.input(width, name)?),
            _ => None,
        };
        Ok(Self {
            table: QMatMul::from_qtensor(table)?,
            inverse,
        })
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let xs = self.table.embedding(ids)?.to_dtype(DType::F32)?;
        match &self.inverse {
            Some(t) => t.invert(&xs),
            None => Ok(xs),
        }
    }
}

/// The draft head of an MTP layer: logits for its `ids` only, the tokens it can propose.
#[derive(Debug, Clone)]
struct DraftHead {
    head: QLinear,
    ids: Vec<u32>,
}

/// An MTP (`nextn`) layer.
#[derive(Debug, Clone)]
struct MtpLayer {
    eh_proj: QLinear,
    enorm: RmsNorm,
    hnorm: RmsNorm,
    embed: Option<Embedding>,
    layer: DecoderLayer,
    head_norm: Option<RmsNorm>,
    head: Option<QLinear>,
    draft_head: Option<DraftHead>,
}

impl MtpLayer {
    fn new<R: Read + Seek>(
        w: &mut Weights<R>,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        il: usize,
    ) -> Result<Self> {
        let p = format!("blk.{il}");
        let n = |s: &str| format!("{p}.nextn.{s}");
        let embed = if w.has(&n("embed_tokens.weight")) {
            Some(Embedding::new(w, &n("embed_tokens.weight"))?)
        } else {
            None
        };
        let head_norm = if w.has(&n("shared_head_norm.weight")) {
            Some(w.norm(&n("shared_head_norm.weight"))?)
        } else {
            None
        };
        let head = if w.has(&n("shared_head_head.weight")) {
            Some(w.linear(&n("shared_head_head"))?)
        } else {
            None
        };
        // A draft head only stands in for the model's own LM head.
        let draft_head = if head.is_none() && w.has(&n("draft_head.weight")) {
            let ids_name = n("draft_head_ids.weight");
            let ids: Vec<u32> = match w.ct.raw_tensor_infos.get(&ids_name) {
                Some(info) => info
                    .read_ints(w.reader, w.ct.tensor_data_offset)?
                    .into_iter()
                    .map(|i| i as u32)
                    .collect(),
                None => w
                    .f32(&ids_name)?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|i| i as u32)
                    .collect(),
            };
            let head = w.linear(&n("draft_head"))?;
            Some(DraftHead { head, ids })
        } else {
            None
        };
        Ok(Self {
            eh_proj: w.linear(&n("eh_proj"))?,
            enorm: w.norm(&n("enorm.weight"))?,
            hnorm: w.norm(&n("hnorm.weight"))?,
            embed,
            layer: DecoderLayer::new(w, cfg, false, rotary, &p)?,
            head_norm,
            head,
            draft_head,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    output: QLinear,
    mtp: Vec<MtpLayer>,
    rotary: Arc<RotaryEmbedding>,
    /// The RoPE position of cache position `p` is `p + rope_delta`, see
    /// [`Model::set_rope_delta`].
    rope_delta: i64,
    device: Device,
}

impl Model {
    /// Loads a `qwen35` GGUF file. The MTP layers are loaded when it has them.
    pub fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let cfg = Config::from_gguf(ct)?;
        Self::new(&cfg, ct, reader, device)
    }

    pub fn new<R: Read + Seek>(
        cfg: &Config,
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut w = Weights {
            ct,
            reader,
            device: device.clone(),
            hadamard: Hadamard::from_gguf(ct, device)?,
            eps: cfg.rms_norm_eps as f32,
        };
        let rotary = Arc::new(
            RotaryEmbedding::new_partial(
                DType::F32,
                cfg.rope_dim,
                cfg.rope_theta,
                cfg.max_position_embeddings,
                device,
            )?
            .with_mrope_section(cfg.mrope_section.clone()),
        );
        let embed = Embedding::new(&mut w, "token_embd.weight")?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("blk.{i}");
                DecoderLayer::new(&mut w, cfg, cfg.recurrent[i], rotary.clone(), &p)
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = w.norm("output_norm.weight")?;
        let output = if w.has("output.weight") {
            w.linear("output")?
        } else {
            w.linear("token_embd")?
        };
        let mtp = (0..cfg.num_mtp_layers)
            .map(|i| MtpLayer::new(&mut w, cfg, rotary.clone(), cfg.num_hidden_layers + i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed,
            layers,
            norm,
            output,
            mtp,
            rotary,
            rope_delta: 0,
            device: device.clone(),
        })
    }

    fn causal_mask(&self, l: usize, offset: usize) -> Result<Option<Tensor>> {
        if l == 1 {
            return Ok(None);
        }
        let mask =
            crate::utils::build_additive_causal_mask(l, offset, None, &self.device, DType::F32)?;
        Ok(Some(mask))
    }

    /// The `(batch, seq, hidden)` final hidden states of `input` `(batch, seq)` after `offset`
    /// already-processed tokens.
    pub fn forward_hidden(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let embeds = self.embed(input)?;
        self.forward_embeds(&embeds, offset, None)
    }

    /// The `(batch, seq, hidden)` token embeddings of `input`.
    pub fn embed(&self, input: &Tensor) -> Result<Tensor> {
        self.embed.forward(input)
    }

    /// Runs `(batch, seq, hidden)` input embeddings after `offset` cached positions, with the
    /// temporal, height and width RoPE positions of images when given; see
    /// [`super::qwen3_5::Model::forward_embeds`].
    pub fn forward_embeds(
        &mut self,
        embeds: &Tensor,
        offset: usize,
        positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<Tensor> {
        let (_, l, _) = embeds.dims3()?;
        let mask = self.causal_mask(l, offset)?;
        let rope = self.rope(l, offset, positions)?;
        let mut xs = embeds.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), &rope)?;
        }
        self.norm.forward(&xs)
    }

    fn rope(&self, l: usize, offset: usize, positions: Option<&[Vec<u32>; 3]>) -> Result<Rope> {
        match positions {
            Some(p) => {
                if p.iter().any(|p| p.len() != l) {
                    candle::bail!("expected {l} positions per axis")
                }
                self.rotary.mrope_cos_sin(p)
            }
            None => {
                let start = offset as i64 + self.rope_delta;
                if start < 0 {
                    candle::bail!("negative RoPE position {start}")
                }
                self.rotary.cos_sin(start as usize, l)
            }
        }
    }

    /// Runs a `(1, seq)` prompt from cleared caches, its image tokens `image_token_id` replaced
    /// by the embeddings of `images` from the vision tower of an `mmproj` file, and returns the
    /// final hidden states. Later steps take [`Model::forward`] at the next cache position.
    pub fn forward_prompt_hidden(
        &mut self,
        ids: &Tensor,
        images: Option<(&VisionModel, Images<'_>)>,
        image_token_id: u32,
    ) -> Result<Tensor> {
        self.clear_kv_cache();
        let embeds = self.embed(ids)?;
        let Some((vision, images)) = images else {
            return self.forward_embeds(&embeds, 0, None);
        };
        let image_embeds = vision.forward(images.pixel_values, images.grids)?;
        let merge = vision.config().spatial_merge_size;
        let (embeds, positions, delta) = prompt_embeds(
            ids,
            &embeds,
            &image_embeds,
            images.grids,
            image_token_id,
            merge,
        )?;
        let hidden = self.forward_embeds(&embeds, 0, Some(&positions))?;
        self.rope_delta = delta;
        Ok(hidden)
    }

    /// Sets the shift between cache positions and RoPE positions after a multimodal prompt; it
    /// is reset by [`Model::clear_kv_cache`].
    pub fn set_rope_delta(&mut self, delta: i64) {
        self.rope_delta = delta
    }

    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }

    /// The LM head applied to normed hidden states.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    /// The `(batch, 1, vocab)` logits of the last position.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let h = self.forward_hidden(input, offset)?;
        self.logits(&h.narrow(1, l - 1, 1)?)
    }

    /// The `(batch, seq, vocab)` logits of every position.
    pub fn forward_all(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.forward_hidden(input, offset)?;
        self.logits(&h)
    }

    /// Clears the KV caches and the recurrent states, to start a new sequence.
    pub fn clear_kv_cache(&mut self) {
        self.rope_delta = 0;
        let layers = self.layers.iter_mut();
        for layer in layers.chain(self.mtp.iter_mut().map(|m| &mut m.layer)) {
            match &mut layer.mixer {
                TokenMixer::Linear(m) => {
                    m.conv_state = None;
                    m.recurrent_state = None;
                }
                TokenMixer::Full(m) => m.kv_cache = None,
            }
        }
    }

    pub fn cache(&self) -> Cache {
        Cache {
            layers: self.layers.iter().map(|l| l.cache()).collect(),
            mtp_layers: self.mtp.iter().map(|m| m.layer.cache()).collect(),
        }
    }

    pub fn set_cache(&mut self, cache: Cache) -> Result<()> {
        if cache.layers.len() != self.layers.len() || cache.mtp_layers.len() != self.mtp.len() {
            candle::bail!("cache of another model")
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.layers) {
            layer.set_cache(c)?
        }
        for (m, c) in self.mtp.iter_mut().zip(cache.mtp_layers) {
            m.layer.set_cache(c)?
        }
        Ok(())
    }

    /// Whether the file has MTP layers.
    pub fn has_mtp(&self) -> bool {
        !self.mtp.is_empty()
    }

    /// Whether the first MTP layer drafts from a trimmed vocabulary.
    pub fn has_draft_head(&self) -> bool {
        self.mtp.first().is_some_and(|m| m.draft_head.is_some())
    }

    /// Runs MTP layer `step % num_mtp_layers` on tokens `ids` `(batch, seq)`, each paired with
    /// `hidden`, the normed hidden state of the position before it, at MTP cache positions from
    /// `offset`. Returns the normed `(batch, seq, hidden)` outputs.
    pub fn mtp_forward_hidden(
        &mut self,
        ids: &Tensor,
        hidden: &Tensor,
        offset: usize,
        step: usize,
    ) -> Result<Tensor> {
        if self.mtp.is_empty() {
            candle::bail!("the file has no MTP layers")
        }
        let (_, l) = ids.dims2()?;
        let mask = self.causal_mask(l, offset)?;
        let rope = self.rope(l, offset, None)?;
        let n = self.mtp.len();
        let embeds = match &self.mtp[step % n].embed {
            Some(e) => e.forward(ids)?,
            None => self.embed.forward(ids)?,
        };
        let m = &mut self.mtp[step % n];
        let e = m.enorm.forward(&embeds)?;
        let h = m.hnorm.forward(hidden)?;
        let xs = Tensor::cat(&[e, h], D::Minus1)?.apply(&m.eh_proj)?;
        let xs = m.layer.forward(&xs, mask.as_ref(), &rope)?;
        match &m.head_norm {
            Some(norm) => norm.forward(&xs),
            None => self.norm.forward(&xs),
        }
    }

    /// The full-vocabulary logits of MTP outputs from step `step`.
    pub fn mtp_logits(&self, hidden: &Tensor, step: usize) -> Result<Tensor> {
        match self
            .mtp
            .get(step % self.mtp.len().max(1))
            .and_then(|m| m.head.as_ref())
        {
            Some(head) => head.forward(hidden),
            None => self.logits(hidden),
        }
    }

    /// Greedy decoding with the MTP layers drafting `drafts` tokens per main-model pass; see
    /// [`generate_mtp`].
    pub fn generate_mtp(
        &mut self,
        prompt: &Tensor,
        max_new: usize,
        drafts: usize,
        on_token: impl FnMut(u32) -> Result<bool>,
    ) -> Result<MtpStats> {
        generate_mtp(self, prompt, max_new, drafts, on_token)
    }
}

impl MtpModel for Model {
    type Cache = Cache;

    fn device(&self) -> &Device {
        &self.device
    }

    fn clear_kv_cache(&mut self) {
        Model::clear_kv_cache(self)
    }

    fn forward_hidden(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        Model::forward_hidden(self, input, offset)
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Model::logits(self, hidden)
    }

    fn mtp_forward_hidden(
        &mut self,
        ids: &Tensor,
        hidden: &Tensor,
        offset: usize,
        step: usize,
    ) -> Result<Tensor> {
        Model::mtp_forward_hidden(self, ids, hidden, offset, step)
    }

    /// The draft head proposes among its own tokens; otherwise the MTP head's logits decide.
    /// [`generate_mtp`] drafts its first token at step 0 and chains on.
    fn draft_token(&self, hidden: &Tensor) -> Result<u32> {
        let argmax = |l: Tensor| l.flatten_all()?.argmax(0)?.to_scalar::<u32>();
        match self.mtp.first().and_then(|m| m.draft_head.as_ref()) {
            Some(d) => Ok(d.ids[argmax(d.head.forward(hidden)?)? as usize]),
            None => argmax(self.mtp_logits(hidden, 0)?),
        }
    }

    fn cache(&self) -> Cache {
        Model::cache(self)
    }

    fn set_cache(&mut self, cache: Cache) -> Result<()> {
        Model::set_cache(self, cache)
    }

    fn set_mtp_cache(&mut self, cache: Cache) -> Result<()> {
        let cache = Cache {
            layers: Model::cache(self).layers,
            mtp_layers: cache.mtp_layers,
        };
        Model::set_cache(self, cache)
    }
}
