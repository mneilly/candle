//! The FLUX.2 transformer, in the diffusers layout (`transformer/` of the checkpoints).

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, LayerNorm, Linear, RmsNorm, VarBuilder};

fn default_patch_size() -> usize {
    1
}

fn default_in_channels() -> usize {
    128
}

fn default_timestep_channels() -> usize {
    256
}

fn default_mlp_ratio() -> f64 {
    3.
}

fn default_axes_dims_rope() -> Vec<usize> {
    vec![32, 32, 32, 32]
}

fn default_rope_theta() -> f64 {
    2000.
}

fn default_eps() -> f64 {
    1e-6
}

fn default_true() -> bool {
    true
}

/// The transformer config, `transformer/config.json` in the checkpoints.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub joint_attention_dim: usize,
    #[serde(default = "default_timestep_channels")]
    pub timestep_guidance_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// Whether the model takes a guidance embedding, as the guidance-distilled FLUX.2 [dev].
    #[serde(default = "default_true")]
    pub guidance_embeds: bool,
}

impl Config {
    fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
}

/// Sinusoidal timestep features, `[cos, sin]` as with diffusers' `flip_sin_to_cos=True`.
fn timestep_features(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let args = t
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&freqs)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

/// `linear_2(silu(linear_1(x)))`, diffusers' `TimestepEmbedding` without biases.
#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(in_dim: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_no_bias(in_dim, dim, vb.pp("linear_1"))?,
            linear_2: linear_no_bias(dim, dim, vb.pp("linear_2"))?,
        })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(&xs.apply(&self.linear_1)?)?.apply(&self.linear_2)
    }
}

/// The shift, scale and gate of one modulated sublayer, each `(batch, 1, dim)`.
#[derive(Debug, Clone)]
struct ModParams {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModParams {
    fn modulate(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }
}

/// `linear(silu(temb))`, split into `sets` shift/scale/gate triples. FLUX.2 shares one
/// modulation across all the blocks of a stream.
#[derive(Debug, Clone)]
struct Modulation {
    linear: Linear,
    sets: usize,
}

impl Modulation {
    fn new(dim: usize, sets: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear_no_bias(dim, dim * 3 * sets, vb.pp("linear"))?,
            sets,
        })
    }

    fn forward(&self, temb: &Tensor) -> Result<Vec<ModParams>> {
        let mods = candle_nn::ops::silu(temb)?
            .apply(&self.linear)?
            .unsqueeze(1)?
            .chunk(3 * self.sets, D::Minus1)?;
        Ok(mods
            .chunks(3)
            .map(|c| ModParams {
                shift: c[0].clone(),
                scale: c[1].clone(),
                gate: c[2].clone(),
            })
            .collect())
    }
}

/// RoPE tables for the given `(seq, axes)` position ids: one block of frequencies per axis, in
/// the interleaved (GPT-J) layout. Returns `(cos, sin)`, each `(seq, head_dim / 2)` in f32.
///
/// The frequencies are computed in f64 on the host, as some devices have no f64 support, and the
/// tables are returned on the device of `ids`.
pub fn rope_tables(ids: &Tensor, axes_dims: &[usize], theta: f64) -> Result<(Tensor, Tensor)> {
    let device = ids.device().clone();
    let ids = ids.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    let mut freqs = Vec::with_capacity(axes_dims.len());
    for (axis, &dim) in axes_dims.iter().enumerate() {
        let inv: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1. / theta.powf(i as f64 / dim as f64))
            .collect();
        let inv = Tensor::from_vec(inv, (1, dim / 2), ids.device())?;
        let pos = ids.narrow(1, axis, 1)?;
        freqs.push(pos.broadcast_mul(&inv)?);
    }
    let freqs = Tensor::cat(&freqs, 1)?;
    Ok((
        freqs.cos()?.to_dtype(DType::F32)?.to_device(&device)?,
        freqs.sin()?.to_dtype(DType::F32)?.to_device(&device)?,
    ))
}

fn apply_rope(xs: &Tensor, pe: &(Tensor, Tensor)) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?.contiguous()?;
    candle_nn::rotary_emb::rope_i(&xs, &pe.0, &pe.1)?.to_dtype(dtype)
}

/// Softmax attention over `(batch, heads, seq, head_dim)` tensors, returning `(batch, seq,
/// heads * head_dim)`.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1. / (q.dim(D::Minus1)? as f64).sqrt();
    let scores = (q.matmul(&k.t()?)? * scale)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs
        .matmul(&v.contiguous()?)?
        .transpose(1, 2)?
        .flatten_from(2)
}

fn layer_norm(dim: usize, eps: f64, vb: &VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, eps))
}

/// `linear_out(silu(gate) * up)` where `linear_in` produces `[gate | up]`.
#[derive(Debug, Clone)]
struct FeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_in: linear_no_bias(dim, 2 * hidden, vb.pp("linear_in"))?,
            linear_out: linear_no_bias(hidden, dim, vb.pp("linear_out"))?,
        })
    }
}

fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let halves = xs.chunk(2, D::Minus1)?;
    candle_nn::ops::silu(&halves[0])? * &halves[1]
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        swiglu(&xs.apply(&self.linear_in)?)?.apply(&self.linear_out)
    }
}

/// Projects to `(batch, heads, seq, head_dim)` queries, keys and values, with QK-norm.
#[derive(Debug, Clone)]
struct Qkv {
    q: Linear,
    k: Linear,
    v: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl Qkv {
    fn forward(&self, xs: &Tensor, heads: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, l, _) = xs.dims3()?;
        let split = |lin: &Linear| -> Result<Tensor> {
            xs.apply(lin)?
                .reshape((b, l, heads, ()))?
                .transpose(1, 2)?
                .contiguous()
        };
        Ok((
            split(&self.q)?.apply(&self.norm_q)?,
            split(&self.k)?.apply(&self.norm_k)?,
            split(&self.v)?,
        ))
    }
}

/// A double-stream block: separate weights for the image and text tokens, joint attention.
#[derive(Debug, Clone)]
struct DoubleBlock {
    img_qkv: Qkv,
    txt_qkv: Qkv,
    to_out: Linear,
    to_add_out: Linear,
    norm: LayerNorm,
    ff: FeedForward,
    ff_context: FeedForward,
    heads: usize,
}

impl DoubleBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let hd = cfg.attention_head_dim;
        let a = vb.pp("attn");
        let hidden = (dim as f64 * cfg.mlp_ratio) as usize;
        let rms = |name: &str| candle_nn::rms_norm(hd, cfg.eps, a.pp(name));
        Ok(Self {
            img_qkv: Qkv {
                q: linear_no_bias(dim, dim, a.pp("to_q"))?,
                k: linear_no_bias(dim, dim, a.pp("to_k"))?,
                v: linear_no_bias(dim, dim, a.pp("to_v"))?,
                norm_q: rms("norm_q")?,
                norm_k: rms("norm_k")?,
            },
            txt_qkv: Qkv {
                q: linear_no_bias(dim, dim, a.pp("add_q_proj"))?,
                k: linear_no_bias(dim, dim, a.pp("add_k_proj"))?,
                v: linear_no_bias(dim, dim, a.pp("add_v_proj"))?,
                norm_q: rms("norm_added_q")?,
                norm_k: rms("norm_added_k")?,
            },
            to_out: linear_no_bias(dim, dim, a.pp("to_out.0"))?,
            to_add_out: linear_no_bias(dim, dim, a.pp("to_add_out"))?,
            norm: layer_norm(dim, cfg.eps, &vb)?,
            ff: FeedForward::new(dim, hidden, vb.pp("ff"))?,
            ff_context: FeedForward::new(dim, hidden, vb.pp("ff_context"))?,
            heads: cfg.num_attention_heads,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[ModParams],
        txt_mod: &[ModParams],
        pe: &(Tensor, Tensor),
    ) -> Result<(Tensor, Tensor)> {
        let txt_len = txt.dim(1)?;
        let img_in = img_mod[0].modulate(&self.norm.forward(img)?)?;
        let txt_in = txt_mod[0].modulate(&self.norm.forward(txt)?)?;
        let (iq, ik, iv) = self.img_qkv.forward(&img_in, self.heads)?;
        let (tq, tk, tv) = self.txt_qkv.forward(&txt_in, self.heads)?;
        // Text tokens first, as in the reference.
        let q = apply_rope(&Tensor::cat(&[&tq, &iq], 2)?, pe)?;
        let k = apply_rope(&Tensor::cat(&[&tk, &ik], 2)?, pe)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;
        let attn = attention(&q, &k, &v)?;
        let txt_attn = attn.narrow(1, 0, txt_len)?.apply(&self.to_add_out)?;
        let img_attn = attn
            .narrow(1, txt_len, attn.dim(1)? - txt_len)?
            .apply(&self.to_out)?;

        let img = (img + img_attn.broadcast_mul(&img_mod[0].gate)?)?;
        let ff = self
            .ff
            .forward(&img_mod[1].modulate(&self.norm.forward(&img)?)?)?;
        let img = (&img + ff.broadcast_mul(&img_mod[1].gate)?)?;

        let txt = (txt + txt_attn.broadcast_mul(&txt_mod[0].gate)?)?;
        let ff = self
            .ff_context
            .forward(&txt_mod[1].modulate(&self.norm.forward(&txt)?)?)?;
        let txt = (&txt + ff.broadcast_mul(&txt_mod[1].gate)?)?;
        Ok((img, txt))
    }
}

/// A single-stream block: attention and MLP in parallel over the joint sequence, with fused
/// input and output projections.
#[derive(Debug, Clone)]
struct SingleBlock {
    to_qkv_mlp: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    to_out: Linear,
    norm: LayerNorm,
    dim: usize,
    mlp_hidden: usize,
    heads: usize,
}

impl SingleBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim();
        let hd = cfg.attention_head_dim;
        let a = vb.pp("attn");
        let mlp_hidden = (dim as f64 * cfg.mlp_ratio) as usize;
        Ok(Self {
            to_qkv_mlp: linear_no_bias(dim, 3 * dim + 2 * mlp_hidden, a.pp("to_qkv_mlp_proj"))?,
            norm_q: candle_nn::rms_norm(hd, cfg.eps, a.pp("norm_q"))?,
            norm_k: candle_nn::rms_norm(hd, cfg.eps, a.pp("norm_k"))?,
            to_out: linear_no_bias(dim + mlp_hidden, dim, a.pp("to_out"))?,
            norm: layer_norm(dim, cfg.eps, &vb)?,
            dim,
            mlp_hidden,
            heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, m: &ModParams, pe: &(Tensor, Tensor)) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let h = m
            .modulate(&self.norm.forward(xs)?)?
            .apply(&self.to_qkv_mlp)?;
        let heads = |i: usize| -> Result<Tensor> {
            h.narrow(2, i * self.dim, self.dim)?
                .reshape((b, l, self.heads, ()))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = apply_rope(&heads(0)?.apply(&self.norm_q)?, pe)?;
        let k = apply_rope(&heads(1)?.apply(&self.norm_k)?, pe)?;
        let attn = attention(&q, &k, &heads(2)?)?;
        let mlp = swiglu(&h.narrow(2, 3 * self.dim, 2 * self.mlp_hidden)?)?;
        let out = Tensor::cat(&[attn, mlp], D::Minus1)?.apply(&self.to_out)?;
        xs + out.broadcast_mul(&m.gate)?
    }
}

/// The FLUX.2 transformer.
#[derive(Debug, Clone)]
pub struct Flux2 {
    timestep_embedder: TimestepEmbedding,
    guidance_embedder: Option<TimestepEmbedding>,
    double_mod_img: Modulation,
    double_mod_txt: Modulation,
    single_mod: Modulation,
    x_embedder: Linear,
    context_embedder: Linear,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: LayerNorm,
    norm_out_linear: Linear,
    proj_out: Linear,
    cfg: Config,
}

impl Flux2 {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.patch_size != 1 {
            candle::bail!("flux2: patch_size {} is not supported", cfg.patch_size)
        }
        if cfg.axes_dims_rope.iter().sum::<usize>() != cfg.attention_head_dim {
            candle::bail!("flux2: the RoPE axes must cover the head dimension")
        }
        let dim = cfg.inner_dim();
        let tc = cfg.timestep_guidance_channels;
        let te = vb.pp("time_guidance_embed");
        let guidance_embedder = if cfg.guidance_embeds {
            Some(TimestepEmbedding::new(tc, dim, te.pp("guidance_embedder"))?)
        } else {
            None
        };
        Ok(Self {
            timestep_embedder: TimestepEmbedding::new(tc, dim, te.pp("timestep_embedder"))?,
            guidance_embedder,
            double_mod_img: Modulation::new(dim, 2, vb.pp("double_stream_modulation_img"))?,
            double_mod_txt: Modulation::new(dim, 2, vb.pp("double_stream_modulation_txt"))?,
            single_mod: Modulation::new(dim, 1, vb.pp("single_stream_modulation"))?,
            x_embedder: linear_no_bias(cfg.in_channels, dim, vb.pp("x_embedder"))?,
            context_embedder: linear_no_bias(
                cfg.joint_attention_dim,
                dim,
                vb.pp("context_embedder"),
            )?,
            double_blocks: (0..cfg.num_layers)
                .map(|i| DoubleBlock::new(cfg, vb.pp("transformer_blocks").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            single_blocks: (0..cfg.num_single_layers)
                .map(|i| SingleBlock::new(cfg, vb.pp("single_transformer_blocks").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            norm_out: layer_norm(dim, cfg.eps, &vb)?,
            norm_out_linear: linear_no_bias(dim, 2 * dim, vb.pp("norm_out.linear"))?,
            proj_out: linear_no_bias(dim, cfg.out_channels(), vb.pp("proj_out"))?,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Predicts the flow-matching velocity.
    ///
    /// - `img`: `(batch, img_seq, in_channels)` packed latents, `img_ids`: `(img_seq, 4)`.
    /// - `txt`: `(batch, txt_seq, joint_attention_dim)` text embeddings, `txt_ids`: `(txt_seq, 4)`.
    /// - `timesteps`: `(batch,)` sigmas in `[0, 1]`.
    /// - `guidance`: `(batch,)` guidance scales, for the guidance-distilled models only.
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = img.dtype();
        let tc = self.cfg.timestep_guidance_channels;
        let t = (timesteps.to_dtype(dtype)? * 1000.)?;
        let mut temb = timestep_features(&t, tc)?
            .to_dtype(dtype)?
            .apply(&self.timestep_embedder)?;
        if let (Some(g), Some(embedder)) = (guidance, &self.guidance_embedder) {
            let g = (g.to_dtype(dtype)? * 1000.)?;
            temb = (temb
                + timestep_features(&g, tc)?
                    .to_dtype(dtype)?
                    .apply(embedder)?)?;
        }
        let img_mod = self.double_mod_img.forward(&temb)?;
        let txt_mod = self.double_mod_txt.forward(&temb)?;
        let single_mod = self.single_mod.forward(&temb)?;

        let mut img = img.apply(&self.x_embedder)?;
        let mut txt = txt.apply(&self.context_embedder)?;
        let ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        let pe = rope_tables(&ids, &self.cfg.axes_dims_rope, self.cfg.rope_theta)?;

        for block in self.double_blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &img_mod, &txt_mod, &pe)?;
        }
        let txt_len = txt.dim(1)?;
        let mut xs = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter() {
            xs = block.forward(&xs, &single_mod[0], &pe)?;
        }
        let xs = xs.narrow(1, txt_len, xs.dim(1)? - txt_len)?;

        let emb = candle_nn::ops::silu(&temb)?.apply(&self.norm_out_linear)?;
        let emb = emb.unsqueeze(1)?.chunk(2, D::Minus1)?;
        let (scale, shift) = (&emb[0], &emb[1]);
        let xs = self
            .norm_out
            .forward(&xs)?
            .broadcast_mul(&(scale + 1.)?)?
            .broadcast_add(shift)?;
        xs.apply(&self.proj_out)
    }
}

/// Position ids of the text tokens: `(0, 0, 0, l)` for each position `l`, as `(seq, 4)` u32.
pub fn text_ids(seq_len: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<u32> = (0..seq_len as u32).flat_map(|l| [0, 0, 0, l]).collect();
    Tensor::from_vec(ids, (seq_len, 4), device)
}

/// Position ids of a `height x width` grid of latent tokens, `(t, h, w, 0)` in row-major order,
/// as `(height * width, 4)` u32.
pub fn image_ids(height: usize, width: usize, t: u32, device: &Device) -> Result<Tensor> {
    let ids: Vec<u32> = (0..height as u32)
        .flat_map(|h| (0..width as u32).flat_map(move |w| [t, h, w, 0]))
        .collect();
    Tensor::from_vec(ids, (height * width, 4), device)
}
