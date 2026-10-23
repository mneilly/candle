//! Next-Concept Modeling (NCM): parallel coarse-to-fine text generation.
//!
//! NCM represents a sequence as a hierarchy of discrete codes produced by a multiscale residual
//! VQ-VAE, then trains a transformer to predict that hierarchy one level at a time. All the codes
//! within a level are predicted in parallel, so generating a sequence takes one forward pass per
//! level rather than one per token.
//!
//! This module implements inference only:
//! - the generator ([`Generator`]), which predicts the codes of level `k + 1` from the
//!   reconstruction built from levels `0..=k`,
//! - the multiscale residual quantizer decode path ([`Quantizer`]),
//! - the VQ-VAE decoder ([`VqDecoder`]), which maps the final latent back to token logits.
//!
//! - 📝 [Paper](https://arxiv.org/abs/2610.14162). Next-Concept Modeling: Fast, Parallel
//!   Coarse-to-Fine Text Generation
//! - 💻 [GitHub](https://github.com/HazyResearch/ncm)
//! - 🤗 [Checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints)

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    conv1d, conv_transpose1d, embedding, linear_b, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Embedding, LayerNorm, Linear, VarBuilder,
};

const NORM_EPS: f64 = 1e-6;

fn default_true() -> bool {
    true
}

fn default_rope_base() -> f64 {
    10000.
}

fn default_mlp_ratio() -> f64 {
    4.
}

fn default_compression_factor() -> usize {
    4
}

fn default_dec_layers() -> usize {
    1
}

fn default_cascade_base() -> usize {
    4
}

fn default_quant_resi_ratio() -> f64 {
    0.5
}

fn default_quant_resi_arch() -> String {
    "blend".to_string()
}

/// A config section that only matters through its `enabled` flag.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Toggle {
    #[serde(default)]
    pub enabled: bool,
}

/// The generator config, the `model` section of the reference eval configs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GeneratorConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub bias: bool,
    pub prefix_len: usize,
    /// One vocabulary size per predicted level, i.e. every level but the first.
    pub vocab_size: Vec<usize>,
    #[serde(default)]
    pub use_level_encoding: bool,
    #[serde(default)]
    pub use_rope: bool,
    #[serde(default = "default_rope_base")]
    pub rope_base: f64,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default)]
    pub raw_token_prefix: bool,
    #[serde(default)]
    pub contextual_prefix: bool,
    #[serde(default)]
    pub prefix_pad_token_id: Option<u32>,
    #[serde(default)]
    pub tie_heads_to_codebook: bool,
    #[serde(default = "default_true")]
    pub tie_heads_learn_scale: bool,
    #[serde(default)]
    pub tie_heads_trainable: bool,
    #[serde(default)]
    pub tie_heads_bias: bool,
    #[serde(default = "default_true")]
    pub untied_head_bias: bool,
    #[serde(default)]
    pub n_kv_heads: Option<usize>,
    // Variants of the reference implementation that are not supported here, kept so that a
    // config using them is rejected rather than silently loaded with the wrong architecture.
    #[serde(default)]
    pub predict_level0: bool,
    #[serde(default)]
    pub query_level0: bool,
    #[serde(default)]
    pub shared_output_head: bool,
    #[serde(default)]
    pub ctx_encoder: Option<Toggle>,
    #[serde(default)]
    pub post_upsample_conv: Option<Toggle>,
    #[serde(default)]
    pub use_swiglu: bool,
    #[serde(default)]
    pub use_geglu: bool,
    #[serde(default)]
    pub use_relu2: bool,
    #[serde(default)]
    pub parallel_block: bool,
    #[serde(default)]
    pub normformer: bool,
    #[serde(default)]
    pub use_layerscale: bool,
    #[serde(default)]
    pub use_rezero: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LevelsConfig {
    #[serde(default)]
    pub use_manual_levels: bool,
    #[serde(default)]
    pub manual_levels: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantResiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_quant_resi_ratio")]
    pub ratio: f64,
    #[serde(default)]
    pub share_mode: usize,
    #[serde(default)]
    pub learnable_ratio: bool,
    #[serde(default = "default_quant_resi_arch")]
    pub arch: String,
}

/// The `vector_quantizer_config` section of the VQ-VAE config.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizerConfig {
    pub levels: LevelsConfig,
    #[serde(default)]
    pub learned_l1_sampling: bool,
    #[serde(default)]
    pub learned_all_sampling: bool,
    #[serde(default)]
    pub learned_l1_cascade: bool,
    #[serde(default = "default_cascade_base")]
    pub learned_l1_cascade_base: usize,
    #[serde(default)]
    pub learned_l1_cascade_norm: bool,
    #[serde(default)]
    pub learned_l1_cascade_norm_final: bool,
    #[serde(default)]
    pub quant_resi: Option<QuantResiConfig>,
}

/// The VQ-VAE config, the `tokenizer.vqvae` section of the reference eval configs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VqVaeConfig {
    pub embed_dim: usize,
    pub in_vocab_size: usize,
    pub context_length: usize,
    /// The number of attention heads in the encoder and decoder blocks, not a depth.
    pub n_layers: usize,
    pub vqvae_vocab_size: Vec<usize>,
    #[serde(default = "default_compression_factor")]
    pub compression_factor: usize,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub enc_layers: usize,
    #[serde(default = "default_dec_layers")]
    pub dec_layers: usize,
    #[serde(default)]
    pub dec_positional: bool,
    pub vector_quantizer_config: QuantizerConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub model: GeneratorConfig,
    pub vqvae: VqVaeConfig,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl Config {
    /// The config of the released TinyStories checkpoint.
    pub fn tinystories() -> Self {
        Self {
            model: GeneratorConfig {
                n_layer: 12,
                n_head: 12,
                n_embd: 768,
                bias: true,
                prefix_len: 1,
                vocab_size: vec![4096; 11],
                use_level_encoding: true,
                use_rope: true,
                rope_base: 100000.,
                use_qk_norm: true,
                mlp_ratio: 4.,
                raw_token_prefix: true,
                contextual_prefix: false,
                prefix_pad_token_id: Some(1),
                tie_heads_to_codebook: true,
                tie_heads_learn_scale: true,
                tie_heads_trainable: false,
                tie_heads_bias: false,
                untied_head_bias: true,
                n_kv_heads: None,
                predict_level0: false,
                query_level0: false,
                shared_output_head: false,
                ctx_encoder: None,
                post_upsample_conv: None,
                use_swiglu: false,
                use_geglu: false,
                use_relu2: false,
                parallel_block: false,
                normformer: false,
                use_layerscale: false,
                use_rezero: false,
            },
            vqvae: VqVaeConfig {
                embed_dim: 384,
                in_vocab_size: 8192,
                context_length: 128,
                n_layers: 6,
                vqvae_vocab_size: vec![4096; 12],
                compression_factor: 1,
                use_qk_norm: true,
                enc_layers: 0,
                dec_layers: 1,
                dec_positional: false,
                vector_quantizer_config: QuantizerConfig {
                    levels: LevelsConfig {
                        use_manual_levels: true,
                        manual_levels: vec![1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128],
                    },
                    learned_l1_sampling: true,
                    learned_all_sampling: false,
                    learned_l1_cascade: true,
                    learned_l1_cascade_base: 4,
                    learned_l1_cascade_norm: false,
                    learned_l1_cascade_norm_final: false,
                    quant_resi: Some(QuantResiConfig {
                        enabled: true,
                        ratio: 0.5,
                        share_mode: 0,
                        learnable_ratio: false,
                        arch: "blend".to_string(),
                    }),
                },
            },
            eos_token_id: 0,
            pad_token_id: 1,
        }
    }

    fn check_supported(&self) -> Result<()> {
        let m = &self.model;
        let unsupported = [
            (m.predict_level0, "predict_level0"),
            (m.query_level0, "query_level0"),
            (m.shared_output_head, "shared_output_head"),
            (
                m.ctx_encoder.as_ref().is_some_and(|c| c.enabled),
                "ctx_encoder",
            ),
            (
                m.post_upsample_conv.as_ref().is_some_and(|c| c.enabled),
                "post_upsample_conv",
            ),
            (
                m.use_swiglu || m.use_geglu || m.use_relu2,
                "gated or relu2 mlp",
            ),
            (m.parallel_block, "parallel_block"),
            (m.normformer, "normformer"),
            (m.use_layerscale || m.use_rezero, "layerscale or rezero"),
            (
                m.n_kv_heads.is_some_and(|n| n != m.n_head),
                "grouped-query attention",
            ),
            (!m.raw_token_prefix, "an encoded (non raw-token) prefix"),
            (
                m.contextual_prefix && self.vqvae.enc_layers > 0,
                "contextual_prefix",
            ),
        ];
        let q = &self.vqvae.vector_quantizer_config;
        let unsupported_vq = [
            (!q.levels.use_manual_levels, "level schedulers"),
            (q.learned_all_sampling, "learned_all_sampling"),
            (
                q.learned_l1_cascade_norm || q.learned_l1_cascade_norm_final,
                "learned_l1_cascade_norm",
            ),
            (
                q.quant_resi
                    .as_ref()
                    .is_some_and(|r| r.enabled && r.arch != "blend"),
                "a quant_resi arch other than blend",
            ),
        ];
        for (is_set, name) in unsupported.iter().chain(unsupported_vq.iter()) {
            if *is_set {
                candle::bail!("ncm: {name} is not supported")
            }
        }
        let n_levels = q.levels.manual_levels.len();
        if self.vqvae.vqvae_vocab_size.len() != n_levels {
            candle::bail!(
                "ncm: {} codebooks for {n_levels} levels",
                self.vqvae.vqvae_vocab_size.len()
            )
        }
        if m.vocab_size.len() + 1 != n_levels {
            candle::bail!(
                "ncm: {} generator heads for {n_levels} levels, expected one per level after the first",
                m.vocab_size.len()
            )
        }
        Ok(())
    }
}

fn layer_norm(size: usize, bias: bool, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    if bias {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, NORM_EPS))
    } else {
        Ok(LayerNorm::new_no_bias(weight, NORM_EPS))
    }
}

/// Rotary embedding tables for the given positions, in the half-split (GPT-NeoX) layout.
fn rope_cos_sin(
    positions: &[u32],
    head_dim: usize,
    base: f64,
    dtype: DType,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1. / base.powf(i as f64 / half as f64) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let positions = Tensor::new(positions, dev)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let freqs = positions.broadcast_mul(&inv_freq)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

/// The matrix of `F.interpolate(mode="linear", align_corners=False)` from `src` to `dst` steps.
fn linear_interp_matrix(src: usize, dst: usize) -> Vec<f32> {
    let mut m = vec![0f32; dst * src];
    let scale = src as f64 / dst as f64;
    for i in 0..dst {
        let pos = (scale * (i as f64 + 0.5) - 0.5).max(0.);
        let i0 = pos.floor() as usize;
        let i1 = if i0 + 1 < src { i0 + 1 } else { i0 };
        let w1 = (pos - i0 as f64) as f32;
        m[i * src + i0] += 1. - w1;
        m[i * src + i1] += w1;
    }
    m
}

/// The matrix of `F.interpolate(mode="area")`, i.e. adaptive average pooling, from `src` to `dst`.
fn area_interp_matrix(src: usize, dst: usize) -> Vec<f32> {
    let mut m = vec![0f32; dst * src];
    for i in 0..dst {
        let start = i * src / dst;
        let end = ((i + 1) * src).div_ceil(dst);
        let w = 1. / (end - start) as f32;
        for j in start..end {
            m[i * src + j] = w;
        }
    }
    m
}

/// `numpy.linspace(start, end, n)`.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

#[derive(Debug, Clone)]
struct SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    qk_norm: Option<(Tensor, Tensor)>,
    n_head: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(
        n_embd: usize,
        n_head: usize,
        bias: bool,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = n_embd / n_head;
        let c_attn = linear_b(n_embd, 3 * n_embd, bias, vb.pp("c_attn"))?;
        let c_proj = linear_b(n_embd, n_embd, bias, vb.pp("c_proj"))?;
        let qk_norm = if qk_norm {
            let q = vb.get(head_dim, "q_norm.weight")?;
            let k = vb.get(head_dim, "k_norm.weight")?;
            Some((q, k))
        } else {
            None
        };
        Ok(Self {
            c_attn,
            c_proj,
            qk_norm,
            n_head,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, t, c) = xs.dims3()?;
        let qkv = self.c_attn.forward(xs)?;
        let split = |i: usize| {
            qkv.narrow(2, i * c, c)?
                .reshape((b, t, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (mut q, mut k, v) = (split(0)?, split(1)?, split(2)?);
        if let Some((q_norm, k_norm)) = &self.qk_norm {
            q = candle_nn::ops::rms_norm(&q, q_norm, NORM_EPS as f32)?;
            k = candle_nn::ops::rms_norm(&k, k_norm, NORM_EPS as f32)?;
        }
        if let Some((cos, sin)) = rope {
            q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
            k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        }
        let scale = 1. / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = match mask {
            Some(mask) => att.broadcast_add(mask)?,
            None => att,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let ys = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))?;
        self.c_proj.forward(&ys)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(n_embd: usize, hidden: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear_b(n_embd, hidden, bias, vb.pp("c_fc"))?;
        let c_proj = linear_b(hidden, n_embd, bias, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu_erf()?)
    }
}

/// A pre-norm transformer block, shared by the generator and the VQ-VAE decoder.
#[derive(Debug, Clone)]
struct Block {
    ln_1: LayerNorm,
    attn: SelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(
        n_embd: usize,
        n_head: usize,
        mlp_hidden: usize,
        bias: bool,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln_1 = layer_norm(n_embd, bias, vb.pp("ln_1"))?;
        let attn = SelfAttention::new(n_embd, n_head, bias, qk_norm, vb.pp("attn"))?;
        let ln_2 = layer_norm(n_embd, bias, vb.pp("ln_2"))?;
        let mlp = Mlp::new(n_embd, mlp_hidden, bias, vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.ln_1.forward(xs)?, mask, rope)?)?;
        &xs + self.mlp.forward(&self.ln_2.forward(&xs)?)?
    }
}

/// The blend residual applied after upsampling a level: `(1 - r) * h + r * conv(h)`.
#[derive(Debug, Clone)]
struct Phi {
    conv: Conv1d,
    ratio: f64,
}

impl Phi {
    /// Applies the residual to a `(batch, length, channels)` tensor.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(1, 2)?;
        let ys = ((&xs * (1. - self.ratio))? + (self.conv.forward(&xs)? * self.ratio)?)?;
        ys.transpose(1, 2)
    }
}

/// The decode path of the multiscale residual vector quantizer.
///
/// Level `i` has `levels[i]` codes. Its code embeddings are upsampled to the full length
/// `levels[-1]`, passed through the level's residual, and summed into the reconstruction.
#[derive(Debug, Clone)]
pub struct Quantizer {
    levels: Vec<usize>,
    codebooks: Vec<Embedding>,
    /// Learned transposed convolutions replacing the interpolation of level 0.
    up_l1: Option<Vec<ConvTranspose1d>>,
    /// Linear interpolation matrices `(full_len, levels[i])`, `None` for the last level.
    up: Vec<Option<Tensor>>,
    /// Area interpolation matrices `(levels[i], full_len)`.
    down: Vec<Tensor>,
    /// The residual used by each level, already resolved from the shared or per-level layout.
    phi: Vec<Option<Phi>>,
}

impl Quantizer {
    fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let q = &cfg.vector_quantizer_config;
        let levels = q.levels.manual_levels.clone();
        let n_levels = levels.len();
        let full = *levels.last().unwrap();
        let dim = cfg.embed_dim;
        let (dev, dtype) = (vb.device().clone(), vb.dtype());

        let codebooks = cfg
            .vqvae_vocab_size
            .iter()
            .enumerate()
            .map(|(i, &vs)| embedding(vs, dim, vb.pp(format!("embeddings.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let up_l1 = if q.learned_l1_sampling {
            let stride = full / levels[0];
            if stride * levels[0] != full {
                candle::bail!(
                    "ncm: learned_l1_sampling needs {full} divisible by {}",
                    levels[0]
                )
            }
            let convt = |stride: usize, vb: VarBuilder| {
                let cfg = ConvTranspose1dConfig {
                    stride,
                    ..Default::default()
                };
                conv_transpose1d(dim, dim, stride, cfg, vb)
            };
            let vb = vb.pp("learned_up_l1");
            let convs = if q.learned_l1_cascade {
                let strides = plan_cascade_strides(stride, q.learned_l1_cascade_base);
                strides
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(i, &s)| convt(s, vb.pp(i)))
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![convt(stride, vb)?]
            };
            Some(convs)
        } else {
            None
        };

        let mut up = Vec::with_capacity(n_levels);
        let mut down = Vec::with_capacity(n_levels);
        for (i, &l) in levels.iter().enumerate() {
            let m = if i + 1 == n_levels {
                None
            } else {
                let m = linear_interp_matrix(l, full);
                Some(Tensor::from_vec(m, (full, l), &dev)?.to_dtype(dtype)?)
            };
            up.push(m);
            let m = area_interp_matrix(full, l);
            down.push(Tensor::from_vec(m, (l, full), &dev)?.to_dtype(dtype)?);
        }

        let phi = match q.quant_resi.as_ref().filter(|r| r.enabled) {
            None => vec![None; n_levels],
            Some(r) => {
                let vb = vb.pp("quant_resi");
                let make = |vb: VarBuilder| -> Result<Option<Phi>> {
                    if !r.learnable_ratio && r.ratio.abs() <= 1e-6 {
                        return Ok(None);
                    }
                    let cfg = Conv1dConfig {
                        padding: 1,
                        ..Default::default()
                    };
                    let ratio = if r.learnable_ratio {
                        let logit = vb.get((), "ratio_logit")?.to_dtype(DType::F64)?;
                        let logit = logit.to_scalar::<f64>()?;
                        1. / (1. + (-logit).exp())
                    } else {
                        r.ratio.abs()
                    };
                    let conv = conv1d(dim, dim, 3, cfg, vb)?;
                    Ok(Some(Phi { conv, ratio }))
                };
                let modules = match r.share_mode {
                    0 => (0..n_levels)
                        .map(|i| make(vb.pp(i)))
                        .collect::<Result<Vec<_>>>()?,
                    1 => vec![make(vb.pp("qresi"))?],
                    n => (0..n)
                        .map(|i| make(vb.pp("qresi_ls").pp(i)))
                        .collect::<Result<Vec<_>>>()?,
                };
                // Mirrors PhiNonShared / PhiPartiallyShared: level i uses the module whose tick
                // is closest to i / (n_levels - 1).
                let k = modules.len();
                let ticks = if k == 4 {
                    linspace(1. / 3. / k as f64, 1. - 1. / 3. / k as f64, k)
                } else {
                    linspace(1. / 2. / k as f64, 1. - 1. / 2. / k as f64, k)
                };
                (0..n_levels)
                    .map(|i| {
                        let at = i as f64 / (n_levels.max(2) - 1) as f64;
                        let mut best = 0;
                        for (j, t) in ticks.iter().enumerate() {
                            if (t - at).abs() < (ticks[best] - at).abs() {
                                best = j
                            }
                        }
                        modules[best].clone()
                    })
                    .collect()
            }
        };

        Ok(Self {
            levels,
            codebooks,
            up_l1,
            up,
            down,
            phi,
        })
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub fn codebook(&self, level: usize) -> &Tensor {
        self.codebooks[level].embeddings()
    }

    /// Embeds, upsamples and applies the residual for the `(batch, levels[level])` codes of a
    /// level, returning its `(batch, full_len, dim)` contribution to the reconstruction.
    pub fn level_contribution(&self, level: usize, codes: &Tensor) -> Result<Tensor> {
        let q = self.codebooks[level].forward(codes)?;
        let q = match (&self.up_l1, &self.up[level]) {
            (Some(convs), _) if level == 0 => {
                let mut xs = q.transpose(1, 2)?;
                for conv in convs {
                    xs = conv.forward(&xs)?;
                }
                xs.transpose(1, 2)?
            }
            (_, Some(up)) => up.broadcast_matmul(&q)?,
            (_, None) => q,
        };
        match &self.phi[level] {
            Some(phi) => phi.forward(&q),
            None => Ok(q),
        }
    }

    /// Area-downsamples a `(batch, full_len, dim)` reconstruction to the length of `level`.
    pub fn downsample(&self, z_hat: &Tensor, level: usize) -> Result<Tensor> {
        if self.levels[level] == *self.levels.last().unwrap() {
            Ok(z_hat.clone())
        } else {
            self.down[level].broadcast_matmul(z_hat)
        }
    }

    /// The cumulative reconstruction after each level for flat `(batch, sum(levels))` codes.
    pub fn reconstructions(&self, codes: &Tensor) -> Result<Vec<Tensor>> {
        let mut offset = 0;
        let mut z_hat: Option<Tensor> = None;
        let mut out = Vec::with_capacity(self.levels.len());
        for (level, &l) in self.levels.iter().enumerate() {
            let q = self.level_contribution(level, &codes.narrow(1, offset, l)?)?;
            let z = match z_hat {
                Some(z) => (z + q)?,
                None => q,
            };
            out.push(z.clone());
            z_hat = Some(z);
            offset += l;
        }
        Ok(out)
    }
}

fn plan_cascade_strides(total: usize, base: usize) -> Vec<usize> {
    let mut s = total;
    let mut strides = vec![];
    while s > 1 {
        let mut f = base.min(s);
        while f > 1 && !s.is_multiple_of(f) {
            f -= 1
        }
        if f <= 1 {
            strides.push(s);
            s = 1
        } else {
            strides.push(f);
            s /= f
        }
    }
    if strides.is_empty() {
        strides.push(1)
    }
    strides
}

/// The VQ-VAE decoder, mapping a latent sequence to logits over the base (BPE) vocabulary.
#[derive(Debug, Clone)]
pub struct VqDecoder {
    up: Option<ConvTranspose1d>,
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    output_layer: Linear,
}

impl VqDecoder {
    fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.embed_dim;
        let up = if cfg.compression_factor > 1 {
            let f = cfg.compression_factor;
            let conv_cfg = ConvTranspose1dConfig {
                stride: f,
                ..Default::default()
            };
            Some(conv_transpose1d(dim, dim, f, conv_cfg, vb.pp("up"))?)
        } else {
            None
        };
        let wpe = if cfg.dec_positional {
            Some(embedding(cfg.context_length, dim, vb.pp("wpe"))?)
        } else {
            None
        };
        let block = |vb| Block::new(dim, cfg.n_layers, 4 * dim, true, cfg.use_qk_norm, vb);
        let mut blocks = vec![block(vb.pp("middle_block"))?];
        for i in 0..cfg.dec_layers.saturating_sub(1) {
            blocks.push(block(vb.pp("extra_blocks").pp(i))?);
        }
        let ln_f = layer_norm(dim, false, vb.pp("ln_f"))?;
        let output_layer = linear_b(dim, cfg.in_vocab_size, false, vb.pp("output_layer"))?;
        Ok(Self {
            up,
            wpe,
            blocks,
            ln_f,
            output_layer,
        })
    }
}

impl Module for VqDecoder {
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut xs = match &self.up {
            Some(up) => up.forward(&z.transpose(1, 2)?)?.transpose(1, 2)?,
            None => z.clone(),
        };
        if let Some(wpe) = &self.wpe {
            let pos = Tensor::arange(0u32, xs.dim(1)? as u32, xs.device())?;
            xs = xs.broadcast_add(&wpe.forward(&pos)?)?;
        }
        for block in self.blocks.iter() {
            xs = block.forward(&xs, None, None)?;
        }
        self.output_layer.forward(&self.ln_f.forward(&xs)?)
    }
}

/// An output head predicting the codes of one level.
#[derive(Debug, Clone)]
enum Head {
    Linear(Linear),
    /// Logits are scaled negative squared distances to the level's codebook entries.
    Tied {
        proj: Linear,
        codebook_t: Tensor,
        codebook_sq: Tensor,
        scale: f64,
        bias: Option<Tensor>,
    },
}

impl Head {
    fn tied(cfg: &GeneratorConfig, codebook: &Tensor, vb: VarBuilder) -> Result<Self> {
        let (vocab, dim) = codebook.dims2()?;
        let proj = linear_b(cfg.n_embd, dim, false, vb.pp("proj"))?;
        let codebook = if cfg.tie_heads_trainable {
            vb.get((vocab, dim), "E")?
        } else {
            codebook.clone()
        };
        // d_scale is computed from the frozen codebook the head was built from.
        let d_scale = codebook
            .to_dtype(DType::F32)?
            .sqr()?
            .sum(1)?
            .mean_all()?
            .to_scalar::<f32>()?
            .max(1e-12) as f64;
        let log_inv_t = if cfg.tie_heads_learn_scale {
            vb.get((), "log_inv_T")?
                .to_dtype(DType::F64)?
                .to_scalar::<f64>()?
        } else {
            0.
        };
        let bias = if cfg.tie_heads_bias {
            Some(vb.get(vocab, "bias")?)
        } else {
            None
        };
        Ok(Self::Tied {
            proj,
            codebook_sq: codebook.sqr()?.sum(1)?,
            codebook_t: codebook.t()?.contiguous()?,
            scale: log_inv_t.exp() / d_scale,
            bias,
        })
    }
}

impl Module for Head {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(xs),
            Self::Tied {
                proj,
                codebook_t,
                codebook_sq,
                scale,
                bias,
            } => {
                let z = proj.forward(xs)?;
                let neg_d2 = (z.broadcast_matmul(codebook_t)? * 2.)?.broadcast_sub(codebook_sq)?;
                let logits = (neg_d2 * *scale)?;
                match bias {
                    Some(b) => logits.broadcast_add(b),
                    None => Ok(logits),
                }
            }
        }
    }
}

/// The NCM generator: given the prefix and the reconstruction from levels `0..=k`, it predicts
/// all the codes of level `k + 1` in one forward pass.
#[derive(Debug, Clone)]
pub struct Generator {
    wte: Linear,
    wle: Option<Embedding>,
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    heads: Vec<Head>,
    /// RoPE tables for each predicted level: the prefix at `0..P`, then the level at `0..l`.
    rope: Vec<(Tensor, Tensor)>,
    prefix_len: usize,
}

impl Generator {
    fn new(
        cfg: &GeneratorConfig,
        quantizer: &Quantizer,
        vq_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (dev, dtype) = (vb.device().clone(), vb.dtype());
        let target_levels = &quantizer.levels()[1..];
        let vbt = vb.pp("transformer");
        let wte = linear_b(vq_dim, cfg.n_embd, cfg.bias, vbt.pp("wte"))?;
        let wle = if cfg.use_level_encoding {
            Some(embedding(target_levels.len(), cfg.n_embd, vbt.pp("wle"))?)
        } else {
            None
        };
        let (wpe, rope) = if cfg.use_rope {
            let head_dim = cfg.n_embd / cfg.n_head;
            let rope = target_levels
                .iter()
                .map(|&l| {
                    let pos: Vec<u32> = (0..cfg.prefix_len as u32).chain(0..l as u32).collect();
                    rope_cos_sin(&pos, head_dim, cfg.rope_base, dtype, &dev)
                })
                .collect::<Result<Vec<_>>>()?;
            (None, rope)
        } else {
            let max_len = target_levels.iter().copied().max().unwrap_or(0);
            let max_len = max_len.max(cfg.prefix_len);
            (Some(embedding(max_len, cfg.n_embd, vb.pp("wpe"))?), vec![])
        };
        let hidden = (cfg.mlp_ratio * cfg.n_embd as f64).round() as usize;
        let blocks = (0..cfg.n_layer)
            .map(|i| {
                let vb = vbt.pp("blocks").pp(i);
                Block::new(
                    cfg.n_embd,
                    cfg.n_head,
                    hidden,
                    cfg.bias,
                    cfg.use_qk_norm,
                    vb,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(cfg.n_embd, cfg.bias, vbt.pp("ln_f"))?;
        let heads = cfg
            .vocab_size
            .iter()
            .enumerate()
            .map(|(i, &vs)| {
                let vb = vb.pp("lm_heads").pp(i);
                if cfg.tie_heads_to_codebook {
                    Head::tied(cfg, quantizer.codebook(i + 1), vb)
                } else {
                    Ok(Head::Linear(linear_b(
                        cfg.n_embd,
                        vs,
                        cfg.untied_head_bias,
                        vb,
                    )?))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            wte,
            wle,
            wpe,
            blocks,
            ln_f,
            heads,
            rope,
            prefix_len: cfg.prefix_len,
        })
    }

    fn positions(&self, len: usize, dev: &Device) -> Result<Option<Tensor>> {
        match &self.wpe {
            Some(wpe) => {
                let pos = Tensor::arange(0u32, len as u32, dev)?;
                Ok(Some(wpe.forward(&pos)?))
            }
            None => Ok(None),
        }
    }

    /// Projects the `(batch, prefix_len, vq_dim)` prefix latents to the model width.
    pub fn embed_prefix(&self, z_prefix: &Tensor) -> Result<Tensor> {
        let xs = self.wte.forward(z_prefix)?;
        match self.positions(self.prefix_len, xs.device())? {
            Some(pos) => xs.broadcast_add(&pos),
            None => Ok(xs),
        }
    }

    /// Returns the `(batch, levels[target + 1], vocab)` logits for predicted level `target`
    /// (zero-based over the levels after the first).
    ///
    /// `prefix` is the embedded prefix and `xs` the `(batch, levels[target + 1], vq_dim)`
    /// reconstruction of the previous levels, downsampled to the length of the target level.
    /// `prefix_pad` is an optional `(batch, prefix_len)` mask, non-zero on padded prefix tokens.
    pub fn level_logits(
        &self,
        prefix: &Tensor,
        xs: &Tensor,
        target: usize,
        prefix_pad: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let (p, dev, dtype) = (self.prefix_len, xs.device(), prefix.dtype());
        let mut xs = self.wte.forward(xs)?;
        if let Some(wle) = &self.wle {
            xs = xs.broadcast_add(&wle.embeddings().i(target)?)?;
        }
        if let Some(pos) = self.positions(l, dev)? {
            xs = xs.broadcast_add(&pos)?;
        }
        let mut xs = Tensor::cat(&[prefix, &xs], 1)?;

        // The prefix only attends to itself, the target level attends to everything.
        let s = p + l;
        let mask: Vec<f32> = (0..s)
            .flat_map(|i| {
                (0..s).map(move |j| {
                    if i < p && j >= p {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mut mask = Tensor::from_vec(mask, (1, 1, s, s), dev)?;
        if let Some(pad) = prefix_pad {
            let neg_inf = Tensor::full(f32::NEG_INFINITY, (b, p), dev)?;
            let pad = pad.ne(0u8)?.where_cond(&neg_inf, &neg_inf.zeros_like()?)?;
            let pad = Tensor::cat(&[pad, Tensor::zeros((b, l), DType::F32, dev)?], 1)?;
            mask = mask.broadcast_add(&pad.reshape((b, 1, 1, s))?)?;
        }
        let mask = mask.to_dtype(dtype)?;

        let rope = self.rope.get(target);
        for block in self.blocks.iter() {
            xs = block.forward(&xs, Some(&mask), rope)?;
        }
        let xs = self.ln_f.forward(&xs.narrow(1, p, l)?)?;
        self.heads[target].forward(&xs)
    }
}

/// The full NCM inference stack: generator, quantizer and VQ-VAE decoder.
#[derive(Debug, Clone)]
pub struct Model {
    generator: Generator,
    encoder_wte: Embedding,
    quantizer: Quantizer,
    decoder: VqDecoder,
    prefix_pad_token_id: Option<u32>,
    prefix_len: usize,
}

impl Model {
    /// `vb_generator` holds the generator weights (`generator.pt`) and `vb_tokenizer` the VQ-VAE
    /// weights (`tokenizer.pt`).
    pub fn new(cfg: &Config, vb_generator: VarBuilder, vb_tokenizer: VarBuilder) -> Result<Self> {
        cfg.check_supported()?;
        let vq = &cfg.vqvae;
        let encoder_wte = embedding(
            vq.in_vocab_size,
            vq.embed_dim,
            vb_tokenizer.pp("encoder.wte"),
        )?;
        let quantizer = Quantizer::new(vq, vb_tokenizer.pp("vq"))?;
        let decoder = VqDecoder::new(vq, vb_tokenizer.pp("decoder"))?;
        let generator = Generator::new(&cfg.model, &quantizer, vq.embed_dim, vb_generator)?;
        Ok(Self {
            generator,
            encoder_wte,
            quantizer,
            decoder,
            prefix_pad_token_id: cfg.model.prefix_pad_token_id,
            prefix_len: cfg.model.prefix_len,
        })
    }

    pub fn levels(&self) -> &[usize] {
        self.quantizer.levels()
    }

    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    pub fn decoder(&self) -> &VqDecoder {
        &self.decoder
    }

    /// Generates the full code hierarchy coarse to fine.
    ///
    /// - `prefix`: `(batch, prefix_len)` base-vocabulary token ids conditioning the generation,
    ///   e.g. the end-of-text token for unconditional generation.
    /// - `level0`: `(batch, levels[0])` codes of the first level.
    /// - `sample`: turns the `(batch, levels[k], vocab)` logits of level `k` into
    ///   `(batch, levels[k])` u32 codes.
    ///
    /// Returns the `(batch, sum(levels))` codes of every level, concatenated.
    pub fn generate<F>(&self, prefix: &Tensor, level0: &Tensor, mut sample: F) -> Result<Tensor>
    where
        F: FnMut(usize, &Tensor) -> Result<Tensor>,
    {
        let prefix = prefix.narrow(1, 0, self.prefix_len)?;
        let prefix_pad = match self.prefix_pad_token_id {
            Some(pad) => Some(prefix.eq(pad)?),
            None => None,
        };
        let prefix_emb = self
            .generator
            .embed_prefix(&self.encoder_wte.forward(&prefix)?)?;

        let levels = self.levels();
        let mut codes = vec![level0.clone()];
        let mut z_hat = self.quantizer.level_contribution(0, level0)?;
        for level in 1..levels.len() {
            let xs = self.quantizer.downsample(&z_hat, level)?;
            let logits =
                self.generator
                    .level_logits(&prefix_emb, &xs, level - 1, prefix_pad.as_ref())?;
            let next = sample(level, &logits)?;
            if level + 1 < levels.len() {
                z_hat = (z_hat + self.quantizer.level_contribution(level, &next)?)?;
            }
            codes.push(next);
        }
        Tensor::cat(&codes, 1)
    }

    /// Decodes flat `(batch, sum(levels))` codes to `(batch, full_len, vocab)` token logits.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let z_hat = self.quantizer.reconstructions(codes)?;
        self.decoder.forward(z_hat.last().unwrap())
    }

    /// Decodes the reconstruction after each level, downsampled to the level's length, to token
    /// logits. This shows the text forming coarse to fine.
    pub fn decode_levels(&self, codes: &Tensor) -> Result<Vec<Tensor>> {
        self.quantizer
            .reconstructions(codes)?
            .iter()
            .enumerate()
            .map(|(level, z)| self.decoder.forward(&self.quantizer.downsample(z, level)?))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_matrices() {
        // Upsampling 2 -> 4 with align_corners=False.
        let m = linear_interp_matrix(2, 4);
        assert_eq!(m, [1., 0., 0.75, 0.25, 0.25, 0.75, 0., 1.]);
        // Area 5 -> 2 pools [0, 3) and [2, 5).
        let m = area_interp_matrix(5, 2);
        let t = 1. / 3.;
        assert_eq!(m, [t, t, t, 0., 0., 0., 0., t, t, t]);
    }

    #[test]
    fn cascade_strides() {
        assert_eq!(plan_cascade_strides(128, 4), [4, 4, 4, 2]);
        assert_eq!(plan_cascade_strides(8, 4), [4, 2]);
        assert_eq!(plan_cascade_strides(1, 4), [1]);
    }
}
