//! Qwen3.5 text model, the architecture of the dense Qwen3.5 and Qwen3.8 checkpoints.
//!
//! Qwen3.5 is a hybrid of two token mixers. Three layers out of four use Gated DeltaNet, a
//! linear-attention recurrence that compresses the prefix into a fixed-size state per head. The
//! remaining layers use full softmax attention with an output gate, QK-norm and partial RoPE.
//!
//! Only the text model is implemented: for the multimodal checkpoints, the vision tower is
//! ignored. The multi-token-prediction (MTP) head is loaded when the checkpoint has one, and
//! drafts tokens for speculative decoding with [`Model::generate_mtp`].
//!
//! - 📝 [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
//! - 🤗 [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B),
//!   [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use crate::utils::repeat_kv;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

/// The token mixer used by a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

fn default_head_dim() -> usize {
    256
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_conv_kernel() -> usize {
    4
}

/// The text config, `text_config` in the config of the multimodal checkpoints.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    /// The number of MTP layers; the MTP head is loaded when this is non-zero and the checkpoint
    /// has its weights.
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
}

#[derive(serde::Deserialize)]
struct MultimodalConfig {
    text_config: Config,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

impl Config {
    /// Parses a `config.json`, either of a text-only checkpoint or of a multimodal one, in which
    /// case the text config is read from `text_config`.
    pub fn from_json(json: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(candle::Error::wrap)?;
        if value.get("text_config").is_some() {
            let cfg: MultimodalConfig =
                serde_json::from_value(value).map_err(candle::Error::wrap)?;
            let mut text = cfg.text_config;
            if let Some(tie) = cfg.tie_word_embeddings {
                text.tie_word_embeddings = tie
            }
            Ok(text)
        } else {
            serde_json::from_value(value).map_err(candle::Error::wrap)
        }
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        match &self.layer_types {
            Some(types) => types.clone(),
            None => {
                let interval = self.full_attention_interval.unwrap_or(4);
                (0..self.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            LayerType::FullAttention
                        } else {
                            LayerType::LinearAttention
                        }
                    })
                    .collect()
            }
        }
    }

    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000.)
    }

    fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .or(self.partial_rotary_factor)
            .unwrap_or(0.25)
    }
}

/// RMSNorm with a zero-centered weight, `x * rsqrt(mean(x^2) + eps) * (1 + w)`, computed in f32.
#[derive(Debug, Clone)]
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub(crate) fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = (vb.get(dim, "weight")?.to_dtype(DType::F32)? + 1.)?;
        Ok(Self {
            weight,
            eps: eps as f32,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        candle_nn::ops::rms_norm(
            &xs.to_dtype(DType::F32)?.contiguous()?,
            &self.weight,
            self.eps,
        )?
        .to_dtype(dtype)
    }
}

/// The gated RMSNorm of the Gated DeltaNet output: `rms_norm(x) * w * act(gate)`.
#[derive(Debug, Clone)]
struct RmsNormGated {
    weight: Tensor,
    eps: f32,
    act: Activation,
}

impl RmsNormGated {
    fn new(dim: usize, eps: f64, act: Activation, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?.to_dtype(DType::F32)?;
        Ok(Self {
            weight,
            eps: eps as f32,
            act,
        })
    }

    fn forward(&self, xs: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = candle_nn::ops::rms_norm(
            &xs.to_dtype(DType::F32)?.contiguous()?,
            &self.weight,
            self.eps,
        )?;
        let gate = gate.to_dtype(DType::F32)?.apply(&self.act)?;
        (xs * gate)?.to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let rotary_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor()) as usize;
        Self::new_partial(
            dtype,
            rotary_dim,
            cfg.rope_theta(),
            cfg.max_position_embeddings,
            dev,
        )
    }

    /// RoPE over the first `rotary_dim` features of each head.
    pub(crate) fn new_partial(
        dtype: DType,
        rotary_dim: usize,
        theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    /// Rotates the first `rotary_dim` features of a `(batch, heads, seq, head_dim)` tensor.
    pub(crate) fn apply(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len = xs.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        self.apply_cos_sin(xs, &cos, &sin)
    }

    /// Like [`Self::apply`], for the positions given as a `(seq,)` u32 tensor.
    pub(crate) fn apply_at(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        self.apply_cos_sin(xs, &cos, &sin)
    }

    fn apply_cos_sin(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let head_dim = xs.dim(D::Minus1)?;
        let rot = xs.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        let rot = candle_nn::rotary_emb::rope(&rot, cos, sin)?;
        if self.rotary_dim == head_dim {
            return Ok(rot);
        }
        let pass = xs.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        Tensor::cat(&[&rot, &pass], D::Minus1)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Full softmax attention with QK-norm, partial RoPE and a sigmoid output gate.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(cfg: &Config, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let (nh, nkv, bias) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.attention_bias,
        );
        // q_proj produces the queries and the output gate, interleaved per head.
        let q_proj = linear_b(h, nh * hd * 2, bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(h, nkv * hd, bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(h, nkv * hd, bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(nh * hd, h, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let hd = self.head_dim;
        let q_gate = self
            .q_proj
            .forward(xs)?
            .reshape((b, l, self.num_heads, 2 * hd))?;
        let q = q_gate.narrow(D::Minus1, 0, hd)?;
        let gate = q_gate
            .narrow(D::Minus1, hd, hd)?
            .reshape((b, l, self.num_heads * hd))?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, l, self.num_kv_heads, hd))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, l, self.num_kv_heads, hd))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;

        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            None => (k, v.contiguous()?),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;
        let scale = 1. / (hd as f64).sqrt();
        let scores = (q.contiguous()?.matmul(&k.t()?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?;
        let ys = probs.to_dtype(v.dtype())?.matmul(&v)?;
        let ys = ys.transpose(1, 2)?.reshape((b, l, self.num_heads * hd))?;
        let ys = (ys * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&ys)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

/// The cached state of one layer.
#[derive(Debug, Clone)]
enum LayerCache {
    Linear {
        conv: Option<Tensor>,
        recurrent: Option<Tensor>,
    },
    Full(Option<(Tensor, Tensor)>),
}

/// A snapshot of every cache of a [`Model`], MTP layers included, to roll a sequence back.
/// Tensors are shared, so taking one is cheap.
#[derive(Debug, Clone)]
pub struct Cache {
    layers: Vec<LayerCache>,
    mtp_layers: Vec<LayerCache>,
}

/// `x / sqrt(sum(x^2) + eps)` over the last dimension.
fn l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    xs.broadcast_div(&norm)
}

/// `log(1 + exp(x))`, computed without overflow.
fn softplus(xs: &Tensor) -> Result<Tensor> {
    let tail = ((xs.abs()?.neg()?.exp()? + 1.)?).log()?;
    xs.relu()? + tail
}

/// The gated delta rule over a sequence, one token at a time.
///
/// `q`, `k`: `(batch, heads, seq, dk)`, `v`: `(batch, heads, seq, dv)`, `g` (log decay) and
/// `beta`: `(batch, heads, seq)`, `state`: `(batch, heads, dk, dv)`. Queries are expected to be
/// normalized and scaled already. Returns the `(batch, heads, seq, dv)` output and the new state.
pub fn recurrent_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq_len = q.dim(2)?;
    let mut state = state.clone();
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(2, t, 1)?;
        let k_t = k.narrow(2, t, 1)?;
        let v_t = v.narrow(2, t, 1)?;
        let decay = g.narrow(2, t, 1)?.exp()?.unsqueeze(D::Minus1)?;
        let beta_t = beta.narrow(2, t, 1)?.unsqueeze(D::Minus1)?;
        state = state.broadcast_mul(&decay)?;
        let kv_mem = k_t.matmul(&state)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t)?;
        state = (state + k_t.transpose(2, 3)?.matmul(&delta)?)?;
        outs.push(q_t.matmul(&state)?);
    }
    Ok((Tensor::cat(&outs, 2)?, state))
}

/// The gated delta rule over a sequence, processed in chunks of `chunk_size` tokens.
///
/// Within a chunk the delta-rule updates are condensed into matmuls through the UT transform,
/// and the state is carried from one chunk to the next. Same arguments and outputs as
/// [`recurrent_gated_delta_rule`].
pub fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    chunk_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, h, seq_len, dk) = q.dims4()?;
    let dv = v.dim(D::Minus1)?;
    let c = chunk_size;
    let dev = q.device();
    let pad = (c - seq_len % c) % c;
    let n = (seq_len + pad) / c;
    // Batch and heads are merged into one dimension, as batched matmuls take at most 4 dims.
    let bh = b * h;
    let pad_seq = |xs: &Tensor| xs.pad_with_zeros(2, 0, pad);
    let q = pad_seq(q)?.reshape((bh, n, c, dk))?;
    let k = pad_seq(k)?.reshape((bh, n, c, dk))?;
    let v = pad_seq(v)?.reshape((bh, n, c, dv))?;
    let beta = pad_seq(beta)?.reshape((bh, n, c, 1))?;
    let g = pad_seq(g)?.reshape((bh, n, c))?;

    let k_beta = k.broadcast_mul(&beta)?;
    let v_beta = v.broadcast_mul(&beta)?;
    let cum = g.cumsum(2)?;
    let cum_exp = cum.exp()?.unsqueeze(D::Minus1)?;

    // pairwise[i, j] = exp(cum_i - cum_j) for j <= i, 0 above the diagonal.
    let upper: Vec<f32> = (0..c)
        .flat_map(|i| (0..c).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
        .collect();
    let upper = Tensor::from_vec(upper, (c, c), dev)?;
    let pairwise = cum
        .unsqueeze(3)?
        .broadcast_sub(&cum.unsqueeze(2)?)?
        .broadcast_add(&upper)?
        .exp()?;

    let k_t = k.transpose(2, 3)?.contiguous()?;
    let ut_system = (k_beta.contiguous()?.matmul(&k_t)? * &pairwise)?;
    let intra = (q.contiguous()?.matmul(&k_t)? * &pairwise)?;
    let decayed_k_beta = k_beta.broadcast_mul(&cum_exp)?;

    // Invert the unit lower-triangular system I + L, L = strict_lower(ut_system). With
    // a = -L nilpotent (a^c = 0), (I - a)^-1 = I + a + ... + a^(c-1) = (I + a)(I + a^2)(I + a^4)...,
    // which takes about 2 log2(c) matmuls.
    let strict_lower = Tensor::tril2(c, DType::F32, dev)?.sub(&Tensor::eye(c, DType::F32, dev)?)?;
    let eye = Tensor::eye(c, DType::F32, dev)?;
    let a = ut_system.broadcast_mul(&strict_lower)?.neg()?;
    let mut inv = a.broadcast_add(&eye)?;
    let mut power = a.matmul(&a)?;
    let mut span = 2;
    while span < c {
        inv = inv.matmul(&power.broadcast_add(&eye)?)?;
        span *= 2;
        if span < c {
            power = power.matmul(&power)?;
        }
    }
    let new_values = inv.matmul(&v_beta.contiguous()?)?;
    let k_cumdecay = inv.matmul(&decayed_k_beta.contiguous()?)?;

    let q = q.broadcast_mul(&cum_exp)?;
    let cum_last = cum.narrow(2, c - 1, 1)?;
    let k = k.broadcast_mul(&cum_last.broadcast_sub(&cum)?.exp()?.unsqueeze(D::Minus1)?)?;
    let chunk_decay = cum_last.exp()?.unsqueeze(D::Minus1)?;

    let mut state = state.reshape((bh, dk, dv))?;
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = |xs: &Tensor| xs.narrow(1, i, 1)?.squeeze(1);
        let v_new = (chunk(&new_values)? - chunk(&k_cumdecay)?.matmul(&state)?)?;
        let inter = chunk(&q)?.contiguous()?.matmul(&state)?;
        outs.push((inter + chunk(&intra)?.matmul(&v_new)?)?);
        let k_i = chunk(&k)?.transpose(1, 2)?.contiguous()?;
        state = (state.broadcast_mul(&chunk(&chunk_decay)?)? + k_i.matmul(&v_new)?)?;
    }
    let out = Tensor::cat(&outs, 1)?
        .narrow(1, 0, seq_len)?
        .reshape((b, h, seq_len, dv))?;
    let state = state.reshape((b, h, dk, dv))?;
    Ok((out, state))
}

/// The shapes of a Gated DeltaNet layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GatedDeltaNetConfig {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
    pub rms_norm_eps: f64,
    /// The activation of the output gate.
    pub gate_act: Activation,
}

impl From<&Config> for GatedDeltaNetConfig {
    fn from(cfg: &Config) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            gate_act: cfg.hidden_act,
        }
    }
}

/// The Gated DeltaNet token mixer.
#[derive(Debug, Clone)]
pub(crate) struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    /// The depthwise causal convolution weight, `(conv_dim, kernel)`.
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: RmsNormGated,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel: usize,
    /// The last `kernel - 1` inputs of the convolution, `(batch, conv_dim, kernel - 1)`.
    conv_state: Option<Tensor>,
    /// The recurrent state, `(batch, num_v_heads, head_k_dim, head_v_dim)` in f32.
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNet {
    pub(crate) fn new(cfg: GatedDeltaNetConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nk, nv) = (cfg.num_k_heads, cfg.num_v_heads);
        let (dk, dv) = (cfg.head_k_dim, cfg.head_v_dim);
        let (key_dim, value_dim) = (nk * dk, nv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.conv_kernel;
        let conv_weight = vb
            .get((conv_dim, 1, kernel), "conv1d.weight")?
            .reshape((conv_dim, kernel))?;
        Ok(Self {
            in_proj_qkv: linear_no_bias(h, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: linear_no_bias(h, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: linear_no_bias(h, nv, vb.pp("in_proj_b"))?,
            in_proj_a: linear_no_bias(h, nv, vb.pp("in_proj_a"))?,
            out_proj: linear_no_bias(value_dim, h, vb.pp("out_proj"))?,
            conv_weight,
            a_log: vb.get(nv, "A_log")?.to_dtype(DType::F32)?,
            dt_bias: vb.get(nv, "dt_bias")?.to_dtype(DType::F32)?,
            norm: RmsNormGated::new(dv, cfg.rms_norm_eps, cfg.gate_act, vb.pp("norm"))?,
            num_k_heads: nk,
            num_v_heads: nv,
            head_k_dim: dk,
            head_v_dim: dv,
            kernel,
            conv_state: None,
            recurrent_state: None,
        })
    }

    /// The causal depthwise convolution followed by SiLU, continuing from the cached inputs.
    fn conv(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, conv_dim, l) = xs.dims3()?;
        let xs = xs.to_dtype(self.conv_weight.dtype())?;
        let state = match &self.conv_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((b, conv_dim, self.kernel - 1), xs.dtype(), xs.device())?,
        };
        let full = Tensor::cat(&[&state, &xs], 2)?;
        let mut ys = full
            .narrow(2, 0, l)?
            .broadcast_mul(&self.conv_weight.narrow(1, 0, 1)?)?;
        for j in 1..self.kernel {
            let w = self.conv_weight.narrow(1, j, 1)?;
            ys = (ys + full.narrow(2, j, l)?.broadcast_mul(&w)?)?;
        }
        self.conv_state = Some(full.narrow(2, l, self.kernel - 1)?.contiguous()?);
        candle_nn::ops::silu(&ys)
    }

    pub(crate) fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let dtype = xs.dtype();
        let (nk, nv, dk, dv) = (
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        );

        let mixed = self.in_proj_qkv.forward(xs)?.transpose(1, 2)?;
        let mixed = self.conv(&mixed)?.transpose(1, 2)?.to_dtype(DType::F32)?;
        let z = self.in_proj_z.forward(xs)?;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(xs)?.to_dtype(DType::F32)?)?;
        let a = self.in_proj_a.forward(xs)?.to_dtype(DType::F32)?;
        let g =
            softplus(&a.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a_log.exp()?.neg()?)?;

        // (batch, seq, heads, dim) -> (batch, heads, seq, dim), with the key heads repeated
        // to match the value heads.
        let heads = |xs: Tensor, n: usize, d: usize| -> Result<Tensor> {
            let xs = xs.reshape((b, l, n, d))?;
            let xs = if n < nv {
                let rep = nv / n;
                xs.unsqueeze(3)?
                    .broadcast_as((b, l, n, rep, d))?
                    .reshape((b, l, nv, d))?
            } else {
                xs
            };
            xs.transpose(1, 2)?.contiguous()
        };
        let q = heads(mixed.narrow(2, 0, nk * dk)?, nk, dk)?;
        let k = heads(mixed.narrow(2, nk * dk, nk * dk)?, nk, dk)?;
        let v = heads(mixed.narrow(2, 2 * nk * dk, nv * dv)?, nv, dv)?;
        let q = (l2norm(&q, 1e-6)? * (1. / (dk as f64).sqrt()))?;
        let k = l2norm(&k, 1e-6)?;
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

        let out = out.transpose(1, 2)?.reshape((b * l * nv, dv))?;
        let z = z.reshape((b * l * nv, dv))?;
        let out = self.norm.forward(&out.to_dtype(dtype)?, &z)?;
        self.out_proj.forward(&out.reshape((b, l, nv * dv))?)
    }

    pub(crate) fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
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
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        layer_type: LayerType,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mixer = match layer_type {
            LayerType::LinearAttention => {
                TokenMixer::Linear(GatedDeltaNet::new(cfg.into(), vb.pp("linear_attn"))?)
            }
            LayerType::FullAttention => {
                TokenMixer::Full(Attention::new(cfg, rotary, vb.pp("self_attn"))?)
            }
        };
        let eps = cfg.rms_norm_eps;
        Ok(Self {
            mixer,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(cfg.hidden_size, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.input_layernorm.forward(xs)?;
        let h = match &mut self.mixer {
            TokenMixer::Linear(m) => m.forward(&h)?,
            TokenMixer::Full(m) => m.forward(&h, mask, offset)?,
        };
        let xs = (xs + h)?;
        let h = self
            .post_attention_layernorm
            .forward(&xs)?
            .apply(&self.mlp)?;
        xs + h
    }

    fn clear_cache(&mut self) {
        match &mut self.mixer {
            TokenMixer::Linear(m) => m.clear_cache(),
            TokenMixer::Full(m) => m.clear_kv_cache(),
        }
    }

    fn cache(&self) -> LayerCache {
        match &self.mixer {
            TokenMixer::Linear(m) => LayerCache::Linear {
                conv: m.conv_state.clone(),
                recurrent: m.recurrent_state.clone(),
            },
            TokenMixer::Full(m) => LayerCache::Full(m.kv_cache.clone()),
        }
    }

    fn set_cache(&mut self, cache: LayerCache) -> Result<()> {
        match (&mut self.mixer, cache) {
            (TokenMixer::Linear(m), LayerCache::Linear { conv, recurrent }) => {
                m.conv_state = conv;
                m.recurrent_state = recurrent;
            }
            (TokenMixer::Full(m), LayerCache::Full(kv)) => m.kv_cache = kv,
            _ => candle::bail!("cache of another layer type"),
        }
        Ok(())
    }
}

/// The multi-token-prediction head: `fc(concat(norm(embed(token)), norm(hidden)))` through
/// full-attention decoder layers and a final norm, sharing the model's embeddings and LM head.
/// Paired with the normed hidden state of a position, the token at the next position predicts
/// the token after it.
#[derive(Debug, Clone)]
struct Mtp {
    fc: Linear,
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Mtp {
    fn new(cfg: &Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let layers = (0..cfg.mtp_num_hidden_layers)
            .map(|i| {
                DecoderLayer::new(
                    cfg,
                    LayerType::FullAttention,
                    rotary.clone(),
                    vb.pp("layers").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fc: linear_no_bias(2 * h, h, vb.pp("fc"))?,
            pre_fc_norm_embedding: RmsNorm::new(h, eps, vb.pp("pre_fc_norm_embedding"))?,
            pre_fc_norm_hidden: RmsNorm::new(h, eps, vb.pp("pre_fc_norm_hidden"))?,
            layers,
            norm: RmsNorm::new(h, eps, vb.pp("norm"))?,
        })
    }
}

/// Counts of a [`Model::generate_mtp`] run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpStats {
    /// Main-model passes after the prompt, one per round of drafts.
    pub rounds: usize,
    /// Tokens drafted by the MTP head.
    pub drafted: usize,
    /// Drafted tokens the main model accepted.
    pub accepted: usize,
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    mtp: Option<Mtp>,
    device: Device,
    dtype: DType,
}

impl Model {
    /// Loads the text model from a text-only checkpoint (weights under `model.`) or from a
    /// multimodal one (weights under `model.language_model.`).
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else {
            vb.pp("model")
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let layer_types = cfg.layer_types();
        if layer_types.len() != cfg.num_hidden_layers {
            candle::bail!(
                "{} layer types for {} layers",
                layer_types.len(),
                cfg.num_hidden_layers
            )
        }
        let layers = layer_types
            .iter()
            .enumerate()
            .map(|(i, &t)| DecoderLayer::new(cfg, t, rotary.clone(), vb_m.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let mtp = if cfg.mtp_num_hidden_layers > 0 && vb.contains_tensor("mtp.fc.weight") {
            Some(Mtp::new(cfg, rotary, vb.pp("mtp"))?)
        } else {
            None
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mtp,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Runs `input` `(batch, seq)` after `offset` already-processed tokens and returns the
    /// `(batch, seq, hidden)` final hidden states.
    pub fn forward_hidden(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let mask = self.causal_mask(l, offset)?;
        let mut xs = self.embed_tokens.forward(input)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), offset)?;
        }
        self.norm.forward(&xs)
    }

    fn causal_mask(&self, l: usize, offset: usize) -> Result<Option<Tensor>> {
        if l == 1 {
            return Ok(None);
        }
        let mask =
            crate::utils::build_additive_causal_mask(l, offset, None, &self.device, self.dtype)?;
        Ok(Some(mask))
    }

    /// Applies the LM head to `(batch, seq, hidden)` normed hidden states.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.apply(&self.lm_head)
    }

    /// Returns the `(batch, 1, vocab)` logits of the last position.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.forward_hidden(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Returns the `(batch, seq, vocab)` logits of every position.
    pub fn forward_all(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_hidden(input, offset)?.apply(&self.lm_head)
    }

    /// Clears the KV caches and the recurrent states, to start a new sequence.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache()
        }
        for layer in self.mtp_layers_mut() {
            layer.clear_cache()
        }
    }

    fn mtp_layers_mut(&mut self) -> &mut [DecoderLayer] {
        match &mut self.mtp {
            Some(mtp) => &mut mtp.layers,
            None => &mut [],
        }
    }

    /// A snapshot of every cache, to return to with [`Model::set_cache`].
    pub fn cache(&self) -> Cache {
        let mtp_layers = match &self.mtp {
            Some(mtp) => mtp.layers.iter().map(|l| l.cache()).collect(),
            None => vec![],
        };
        Cache {
            layers: self.layers.iter().map(|l| l.cache()).collect(),
            mtp_layers,
        }
    }

    /// Returns every cache to a snapshot taken with [`Model::cache`].
    pub fn set_cache(&mut self, cache: Cache) -> Result<()> {
        if cache.layers.len() != self.layers.len()
            || cache.mtp_layers.len() != self.mtp_layers_mut().len()
        {
            candle::bail!("cache of another model")
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.layers) {
            layer.set_cache(c)?
        }
        for (layer, c) in self.mtp_layers_mut().iter_mut().zip(cache.mtp_layers) {
            layer.set_cache(c)?
        }
        Ok(())
    }

    /// Whether the checkpoint has an MTP head.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Runs the MTP head on tokens `ids` `(batch, seq)`, each paired with `hidden`
    /// `(batch, seq, hidden)`, the normed hidden state of the position before it, at MTP cache
    /// positions from `offset`. Draft step `step` uses MTP layer `step % mtp_num_hidden_layers`.
    /// Returns the normed `(batch, seq, hidden)` outputs: their logits predict the token after
    /// each of `ids`, and an output paired with the drafted token chains the next draft.
    pub fn mtp_forward_hidden(
        &mut self,
        ids: &Tensor,
        hidden: &Tensor,
        offset: usize,
        step: usize,
    ) -> Result<Tensor> {
        let (_, l) = ids.dims2()?;
        let mask = self.causal_mask(l, offset)?;
        let embeds = self.embed_tokens.forward(ids)?;
        let Some(mtp) = &mut self.mtp else {
            candle::bail!("the checkpoint has no MTP head")
        };
        let e = mtp.pre_fc_norm_embedding.forward(&embeds)?;
        let h = mtp.pre_fc_norm_hidden.forward(hidden)?;
        let xs = Tensor::cat(&[e, h], D::Minus1)?.apply(&mtp.fc)?;
        let n = mtp.layers.len();
        let xs = mtp.layers[step % n].forward(&xs, mask.as_ref(), offset)?;
        mtp.norm.forward(&xs)
    }

    /// Greedy decoding of a `(1, seq)` prompt with the MTP head drafting `drafts` tokens per
    /// main-model pass, which verifies them all at once. The tokens are those of plain greedy
    /// decoding, up to rounding. Starts from cleared caches; `on_token` receives each new token
    /// and returns whether to go on, and generation stops after `max_new` tokens.
    ///
    /// A rejected draft leaves the recurrent states past the accepted tokens, so the main model
    /// returns to its snapshot and runs the accepted tokens again.
    pub fn generate_mtp(
        &mut self,
        prompt: &Tensor,
        max_new: usize,
        drafts: usize,
        mut on_token: impl FnMut(u32) -> Result<bool>,
    ) -> Result<MtpStats> {
        if drafts == 0 {
            candle::bail!("generate_mtp needs at least one draft per round")
        }
        let argmax = |logits: &Tensor| -> Result<Vec<u32>> {
            logits.squeeze(0)?.argmax(D::Minus1)?.to_vec1::<u32>()
        };
        let dev = self.device.clone();
        let p = prompt.dim(1)?;
        let mut stats = MtpStats::default();
        if max_new == 0 {
            return Ok(stats);
        }
        self.clear_kv_cache();
        let hidden = self.forward_hidden(prompt, 0)?;
        let mut next = argmax(&self.logits(&hidden.narrow(1, p - 1, 1)?)?)?[0];
        let mut generated = 1;
        if !on_token(next)? || generated == max_new {
            return Ok(stats);
        }
        // The MTP pairs yet to enter its cache: each token with the hidden state before it.
        let mut pending_ids = Tensor::cat(
            &[prompt.narrow(1, 1, p - 1)?, Tensor::new(&[[next]], &dev)?],
            1,
        )?;
        let mut pending_hidden = hidden;
        let (mut pos, mut mtp_pos) = (p, 0);
        loop {
            // Draft. The pending pairs hold the main model's hidden states and are kept; the
            // chained drafts are dropped after verification.
            let n = pending_ids.dim(1)?;
            let g = self.mtp_forward_hidden(&pending_ids, &pending_hidden, mtp_pos, 0)?;
            mtp_pos += n;
            let committed = self.cache();
            let mut g = g.narrow(1, n - 1, 1)?;
            let mut draft = vec![argmax(&self.logits(&g)?)?[0]];
            for step in 1..drafts {
                let ids = Tensor::new(&[[draft[step - 1]]], &dev)?;
                g = self.mtp_forward_hidden(&ids, &g, mtp_pos + step - 1, step)?;
                draft.push(argmax(&self.logits(&g)?)?[0]);
            }
            stats.drafted += drafts;
            stats.rounds += 1;

            // Verify every draft in one pass.
            let mut input = vec![next];
            input.extend(&draft);
            let ids = Tensor::new(input.as_slice(), &dev)?.unsqueeze(0)?;
            let h = self.forward_hidden(&ids, pos)?;
            let target = argmax(&self.logits(&h)?)?;
            let accepted = draft
                .iter()
                .zip(&target)
                .take_while(|(d, t)| d == t)
                .count();
            stats.accepted += accepted;
            let mut tokens = draft[..accepted].to_vec();
            tokens.push(target[accepted]);
            for &t in tokens.iter() {
                generated += 1;
                if !on_token(t)? || generated == max_new {
                    return Ok(stats);
                }
            }

            let h = if accepted < drafts {
                // Back to the state before verification, then through the accepted tokens.
                self.set_cache(committed)?;
                let ids = Tensor::new(&input[..accepted + 1], &dev)?.unsqueeze(0)?;
                self.forward_hidden(&ids, pos)?
            } else {
                let mut cache = committed;
                cache.layers = self.cache().layers;
                self.set_cache(cache)?;
                h
            };
            pending_ids = Tensor::new(tokens.as_slice(), &dev)?.unsqueeze(0)?;
            pending_hidden = h.narrow(1, 0, accepted + 1)?;
            pos += accepted + 1;
            next = target[accepted];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn chunked_matches_recurrent() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, l, dk, dv) = (2, 3, 150, 16, 24);
        let q = l2norm(&Tensor::randn(0f32, 1., (b, h, l, dk), &dev)?, 1e-6)?;
        let q = (q * (1. / (dk as f64).sqrt()))?;
        let k = l2norm(&Tensor::randn(0f32, 1., (b, h, l, dk), &dev)?, 1e-6)?;
        let v = Tensor::randn(0f32, 1., (b, h, l, dv), &dev)?;
        let g = softplus(&Tensor::randn(0f32, 1., (b, h, l), &dev)?)?.neg()?;
        let beta = candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1., (b, h, l), &dev)?)?;
        let state = Tensor::randn(0f32, 1., (b, h, dk, dv), &dev)?;

        let (out_r, state_r) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, &state)?;
        for chunk_size in [16, 64] {
            let (out_c, state_c) =
                chunk_gated_delta_rule(&q, &k, &v, &g, &beta, &state, chunk_size)?;
            assert!(max_abs_diff(&out_r, &out_c)? < 1e-4);
            assert!(max_abs_diff(&state_r, &state_c)? < 1e-4);
        }
        Ok(())
    }
}
