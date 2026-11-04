//! The multimodal Qwen3.5 and Qwen3.8 models: the vision tower and images in the prompt.
//!
//! Images are split into patches of `patch_size` pixels, each repeated over the
//! `temporal_patch_size` frames of a video patch, and grouped by blocks of
//! `spatial_merge_size x spatial_merge_size` patches. The vision tower adds a learned position
//! grid, resampled bilinearly to the image, runs bidirectional attention with 2D RoPE within each
//! image, and merges each block into one token of the language model.
//!
//! Image tokens take multimodal RoPE positions: temporal, height and width indices starting at
//! the position of the image, and the text after an image resumes past its largest side. The
//! vision tower is optional: without it the model runs text, and its weights are never read.
//!
//! - 🤗 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

use super::qwen3_5;
use crate::models::with_tracing::{linear, Linear};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, Activation, LayerNorm, VarBuilder};

fn default_rope_theta() -> f64 {
    10_000.
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: Activation,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub rope_parameters: Option<VisionRopeParameters>,
}

impl VisionConfig {
    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(default_rope_theta(), |r| r.rope_theta)
    }
}

/// The `(t, h, w)` patch grid of an image, `t` frames of `h x w` patches.
pub type Grid = (usize, usize, usize);

/// The images of a prompt: the pixel values of [`ImageProcessor::preprocess`], concatenated, and
/// their grids in prompt order.
#[derive(Debug, Clone, Copy)]
pub struct Images<'a> {
    pub pixel_values: &'a Tensor,
    pub grids: &'a [Grid],
}

/// A multimodal `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub text: qwen3_5::Config,
    pub vision: VisionConfig,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

#[derive(serde::Deserialize)]
struct RawConfig {
    vision_config: VisionConfig,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl Config {
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json).map_err(candle::Error::wrap)?;
        Ok(Self {
            text: qwen3_5::Config::from_json(json)?,
            vision: raw.vision_config,
            image_token_id: raw.image_token_id,
            vision_start_token_id: raw.vision_start_token_id,
            vision_end_token_id: raw.vision_end_token_id,
        })
    }
}

/// The `(height, width)` closest to an image's, in multiples of `factor` (the patch size times the
/// merge size), with `height * width` within `min_pixels..=max_pixels` (Qwen2-VL's
/// `smart_resize`).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200. {
        candle::bail!("the aspect ratio of a {height}x{width} image exceeds 200")
    }
    let f = factor as f64;
    let round = |x: f64| (x / f).round_ties_even() * f;
    let (mut hb, mut wb) = (round(h), round(w));
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// The pixel normalization and patch layout of the image processor.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl ImageProcessor {
    pub fn new(cfg: &VisionConfig) -> Self {
        Self {
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            merge_size: cfg.spatial_merge_size,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }

    /// The multiple of which image sides must be.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// The `(patches, channels * temporal_patch_size * patch_size^2)` pixel values and the
    /// `(t, h, w)` patch grid of a `(height, width, 3)` u8 RGB image already resized to multiples
    /// of [`ImageProcessor::factor`].
    pub fn preprocess(&self, image: &Tensor) -> Result<(Tensor, Grid)> {
        let (height, width, c) = image.dims3()?;
        let f = self.factor();
        if c != 3 || height % f != 0 || width % f != 0 {
            candle::bail!(
                "expected a (h, w, 3) image with sides multiple of {f}, got {height}x{width}x{c}"
            )
        }
        let dev = image.device();
        let mean = Tensor::new(&self.image_mean, dev)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&self.image_std, dev)?.reshape((3, 1, 1))?;
        let x = (image.permute((2, 0, 1))?.to_dtype(DType::F32)? / 255.)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        let (gh, gw) = (height / p, width / p);
        // (c, gh/m, m, p, gw/m, m, p) -> (gh/m, gw/m, m, m, c, p, p), then each patch repeated
        // over the temporal patch.
        let x = x
            .reshape(vec![3, gh / m, m, p, gw / m, m, p])?
            .permute(vec![1, 4, 2, 5, 0, 3, 6])?
            .contiguous()?
            .reshape((gh * gw, 3, 1, p * p))?;
        let x = x.broadcast_as((gh * gw, 3, t, p * p))?.contiguous()?;
        Ok((x.reshape((gh * gw, 3 * t * p * p))?, (1, gh, gw)))
    }
}

#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    act: Activation,
    num_heads: usize,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            norm1: layer_norm(h, 1e-6, vb.pp("norm1"))?,
            norm2: layer_norm(h, 1e-6, vb.pp("norm2"))?,
            qkv: linear(h, 3 * h, vb.pp("attn.qkv"))?,
            proj: linear(h, h, vb.pp("attn.proj"))?,
            fc1: linear(h, cfg.intermediate_size, vb.pp("mlp.linear_fc1"))?,
            fc2: linear(cfg.intermediate_size, h, vb.pp("mlp.linear_fc2"))?,
            act: cfg.hidden_act,
            num_heads: cfg.num_heads,
        })
    }

    /// `xs`: `(seq, hidden)` patches of the images whose lengths are `segments`.
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        segments: &[usize],
    ) -> Result<Tensor> {
        let (seq, hidden) = xs.dims2()?;
        let nh = self.num_heads;
        let hd = hidden / nh;
        let qkv = self
            .qkv
            .forward(&self.norm1.forward(xs)?)?
            .reshape((seq, 3, nh, hd))?
            .permute((1, 2, 0, 3))?
            .to_dtype(DType::F32)?;
        let rope =
            |x: Tensor| candle_nn::rotary_emb::rope(&x.contiguous()?.unsqueeze(0)?, cos, sin);
        let q = rope(qkv.get(0)?)?;
        let k = rope(qkv.get(1)?)?;
        let v = qkv.get(2)?.unsqueeze(0)?;
        // Attention within each image.
        let mut outs = Vec::with_capacity(segments.len());
        let mut start = 0;
        for &len in segments {
            let (q, k, v) = (
                q.narrow(2, start, len)?.contiguous()?,
                k.narrow(2, start, len)?.contiguous()?,
                v.narrow(2, start, len)?.contiguous()?,
            );
            let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            outs.push(probs.matmul(&v)?);
            start += len;
        }
        let attn = Tensor::cat(&outs, 2)?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((seq, hidden))?
            .to_dtype(xs.dtype())?;
        let xs = (xs + self.proj.forward(&attn)?)?;
        let h = self.fc2.forward(
            &self
                .fc1
                .forward(&self.norm2.forward(&xs)?)?
                .apply(&self.act)?,
        )?;
        xs + h
    }
}

/// The vision tower and the patch merger.
#[derive(Debug, Clone)]
pub struct VisionModel {
    patch_embed: Linear,
    pos_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger_norm: LayerNorm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    inv_freq: Vec<f32>,
    cfg: VisionConfig,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (h, c, p, t) = (
            cfg.hidden_size,
            cfg.in_channels,
            cfg.patch_size,
            cfg.temporal_patch_size,
        );
        // The patch convolution has its stride equal to its kernel: a linear map of each patch.
        let w = vb
            .pp("patch_embed.proj")
            .get((h, c, t, p, p), "weight")?
            .reshape((h, c * t * p * p))?;
        let b = vb.pp("patch_embed.proj").get(h, "bias")?;
        let merged = h * cfg.spatial_merge_size.pow(2);
        let head_dim = h / cfg.num_heads;
        // Half of each head rotates with the height, half with the width.
        let spatial = head_dim / 2;
        let inv_freq = (0..spatial)
            .step_by(2)
            .map(|i| 1. / cfg.rope_theta().powf(i as f64 / spatial as f64) as f32)
            .collect();
        Ok(Self {
            patch_embed: Linear::from_weights(w, Some(b)),
            pos_embed: vb.get((cfg.num_position_embeddings, h), "pos_embed.weight")?,
            blocks: (0..cfg.depth)
                .map(|i| VisionBlock::new(cfg, vb.pp("blocks").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            merger_norm: layer_norm(h, 1e-6, vb.pp("merger.norm"))?,
            merger_fc1: linear(merged, merged, vb.pp("merger.linear_fc1"))?,
            merger_fc2: linear(merged, cfg.out_hidden_size, vb.pp("merger.linear_fc2"))?,
            inv_freq,
            cfg: cfg.clone(),
        })
    }

    /// The `(row, col)` of each patch of a `(t, h, w)` grid, in merge-block order.
    fn patch_coords(&self, (t, h, w): Grid) -> Vec<(usize, usize)> {
        let m = self.cfg.spatial_merge_size;
        let mut coords = Vec::with_capacity(t * h * w);
        for _ in 0..t {
            for br in 0..h / m {
                for bc in 0..w / m {
                    for ir in 0..m {
                        for ic in 0..m {
                            coords.push((br * m + ir, bc * m + ic))
                        }
                    }
                }
            }
        }
        coords
    }

    /// The learned position grid resampled bilinearly, corners aligned, to each patch.
    fn position_embeddings(&self, grids: &[Grid]) -> Result<Tensor> {
        let side = (self.cfg.num_position_embeddings as f64).sqrt() as usize;
        // The two taps and weights of `index` on an axis of `size` patches.
        let taps = |index: usize, size: usize| {
            let src = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
            let floor = src.floor();
            let lo = (floor as usize).min(side - 1);
            let hi = (floor as usize + 1).min(side - 1);
            [(lo, 1. - (src - floor)), (hi, src - floor)]
        };
        let (mut idx, mut wts) = (vec![], vec![]);
        for &(t, h, w) in grids {
            for (r, c) in self.patch_coords((t, h, w)) {
                for (hi, hw) in taps(r, h) {
                    for (wi, ww) in taps(c, w) {
                        idx.push((hi * side + wi) as u32);
                        wts.push(hw * ww);
                    }
                }
            }
        }
        let n = idx.len() / 4;
        let dev = self.pos_embed.device();
        let rows = self
            .pos_embed
            .index_select(&Tensor::new(idx, dev)?, 0)?
            .reshape((n, 4, self.cfg.hidden_size))?;
        let wts = Tensor::from_vec(wts, (n, 4, 1), dev)?.to_dtype(rows.dtype())?;
        rows.broadcast_mul(&wts)?.sum(1)
    }

    /// The cos and sin of the 2D RoPE: the first half of the frequencies rotate with the row
    /// and the second with the column.
    fn rope(&self, grids: &[Grid], dev: &Device) -> Result<(Tensor, Tensor)> {
        let q = self.inv_freq.len();
        let mut freqs = vec![];
        for &g in grids {
            for (r, c) in self.patch_coords(g) {
                freqs.extend(self.inv_freq.iter().map(|f| r as f32 * f));
                freqs.extend(self.inv_freq.iter().map(|f| c as f32 * f));
            }
        }
        let n = freqs.len() / (2 * q);
        let freqs = Tensor::from_vec(freqs, (n, 2 * q), dev)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// The `(tokens, out_hidden_size)` merged image embeddings of the `pixel_values` of
    /// [`ImageProcessor::preprocess`], concatenated over the images of `grids`.
    pub fn forward(&self, pixel_values: &Tensor, grids: &[Grid]) -> Result<Tensor> {
        let dtype = self.pos_embed.dtype();
        let xs = self.patch_embed.forward(&pixel_values.to_dtype(dtype)?)?;
        let xs = (xs + self.position_embeddings(grids)?)?;
        let (cos, sin) = self.rope(grids, xs.device())?;
        // Attention runs within each frame of each image.
        let segments: Vec<usize> = grids
            .iter()
            .flat_map(|&(t, h, w)| std::iter::repeat_n(h * w, t))
            .collect();
        let mut xs = xs;
        for block in self.blocks.iter() {
            xs = block.forward(&xs, &cos, &sin, &segments)?;
        }
        let merged = self.cfg.hidden_size * self.cfg.spatial_merge_size.pow(2);
        let xs = self.merger_norm.forward(&xs)?;
        let xs = xs.reshape((xs.elem_count() / merged, merged))?;
        self.merger_fc2
            .forward(&self.merger_fc1.forward(&xs)?.gelu_erf()?)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Loads the vision tower of a llama.cpp `mmproj` GGUF file (projector `qwen3vl_merger`),
    /// with the image normalization it stores.
    pub fn from_mmproj<R: std::io::Read + std::io::Seek>(
        ct: &candle::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<(Self, ImageProcessor)> {
        let md = &ct.metadata;
        let get = |k: &str| match md.get(k) {
            Some(v) => Ok(v),
            None => candle::bail!("cannot find {k} in the mmproj metadata"),
        };
        let int = |k: &str| -> Result<usize> { Ok(get(k)?.to_u64()? as usize) };
        let projector = get("clip.projector_type")
            .or_else(|_| get("clip.vision.projector_type"))?
            .to_string()?;
        if projector != "qwen3vl_merger" {
            candle::bail!("expected a qwen3vl_merger projector, got {projector}")
        }
        if let Some(v) = md.get("clip.vision.is_deepstack_layers") {
            if v.to_vec()?.iter().any(|v| v.to_bool().unwrap_or(false)) {
                candle::bail!("deepstack layers are not supported")
            }
        }
        let floats = |k: &str| -> Result<[f32; 3]> {
            let v = get(k)?.to_vec()?;
            if v.len() != 3 {
                candle::bail!("expected 3 values for {k}")
            }
            Ok([v[0].to_f32()?, v[1].to_f32()?, v[2].to_f32()?])
        };
        let patch_size = int("clip.vision.patch_size")?;
        let side = int("clip.vision.image_size")? / patch_size;
        let cfg = VisionConfig {
            depth: int("clip.vision.block_count")?,
            hidden_size: int("clip.vision.embedding_length")?,
            hidden_act: Activation::GeluPytorchTanh,
            intermediate_size: int("clip.vision.feed_forward_length")?,
            num_heads: int("clip.vision.attention.head_count")?,
            in_channels: 3,
            patch_size,
            spatial_merge_size: int("clip.vision.spatial_merge_size")?,
            temporal_patch_size: 2,
            out_hidden_size: int("clip.vision.projection_dim")?,
            num_position_embeddings: side * side,
            rope_parameters: None,
        };
        let mut processor = ImageProcessor::new(&cfg);
        processor.image_mean = floats("clip.vision.image_mean")?;
        processor.image_std = floats("clip.vision.image_std")?;

        // The tensors under the names of the transformers checkpoint.
        let mut t = std::collections::HashMap::new();
        let mut read =
            |name: &str| -> Result<Tensor> { ct.tensor(reader, name, device)?.dequantize(device) };
        // The 3D patch convolution is stored as one 2D convolution per frame.
        let frames = [read("v.patch_embd.weight")?, read("v.patch_embd.weight.1")?];
        t.insert(
            "patch_embed.proj.weight".to_string(),
            Tensor::stack(&frames, 2)?,
        );
        t.insert(
            "patch_embed.proj.bias".to_string(),
            read("v.patch_embd.bias")?,
        );
        t.insert(
            "pos_embed.weight".to_string(),
            read("v.position_embd.weight")?,
        );
        for i in 0..cfg.depth {
            for (g, h) in [
                ("ln1", "norm1"),
                ("ln2", "norm2"),
                ("attn_qkv", "attn.qkv"),
                ("attn_out", "attn.proj"),
                ("ffn_up", "mlp.linear_fc1"),
                ("ffn_down", "mlp.linear_fc2"),
            ] {
                for p in ["weight", "bias"] {
                    t.insert(
                        format!("blocks.{i}.{h}.{p}"),
                        read(&format!("v.blk.{i}.{g}.{p}"))?,
                    );
                }
            }
        }
        for (g, h) in [
            ("v.post_ln", "merger.norm"),
            ("mm.0", "merger.linear_fc1"),
            ("mm.2", "merger.linear_fc2"),
        ] {
            for p in ["weight", "bias"] {
                t.insert(format!("{h}.{p}"), read(&format!("{g}.{p}"))?);
            }
        }
        let vb = VarBuilder::from_tensors(t, DType::F32, device);
        Ok((Self::new(&cfg, vb)?, processor))
    }
}

/// The multimodal RoPE positions of a prompt whose image tokens stand for the images of `grids`,
/// in order, each a run of `t * h * w / merge^2` tokens, and the shift between cache and RoPE
/// positions after it.
pub fn rope_positions(
    ids: &[u32],
    grids: &[Grid],
    image_token_id: u32,
    merge: usize,
) -> Result<([Vec<u32>; 3], i64)> {
    let mut pos: [Vec<u32>; 3] = [vec![], vec![], vec![]];
    let mut grids = grids.iter();
    let mut next = 0u32;
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != image_token_id {
            for p in pos.iter_mut() {
                p.push(next)
            }
            next += 1;
            i += 1;
            continue;
        }
        let Some(&(t, h, w)) = grids.next() else {
            candle::bail!("more images in the prompt than given")
        };
        let (h, w) = (h / merge, w / merge);
        let n = t * h * w;
        if ids[i..].len() < n || ids[i..i + n].iter().any(|&id| id != image_token_id) {
            candle::bail!("an image of {n} tokens is not a run of {n} image tokens")
        }
        for tt in 0..t {
            for r in 0..h {
                for c in 0..w {
                    pos[0].push(next + tt as u32);
                    pos[1].push(next + r as u32);
                    pos[2].push(next + c as u32);
                }
            }
        }
        next += h.max(w) as u32;
        i += n;
    }
    if grids.next().is_some() {
        candle::bail!("fewer images in the prompt than given")
    }
    Ok((pos, next as i64 - ids.len() as i64))
}

/// The inputs of a `(1, seq)` prompt with images: its token embeddings `(1, seq, hidden)` with
/// the image tokens replaced by the `(tokens, hidden)` image embeddings, in order, their RoPE
/// positions and the shift of the later positions, see [`rope_positions`].
pub fn prompt_embeds(
    ids: &Tensor,
    embeds: &Tensor,
    image_embeds: &Tensor,
    grids: &[Grid],
    image_token_id: u32,
    merge: usize,
) -> Result<(Tensor, [Vec<u32>; 3], i64)> {
    let tokens = ids.squeeze(0)?.to_vec1::<u32>()?;
    let (positions, delta) = rope_positions(&tokens, grids, image_token_id, merge)?;
    let n_image = tokens.iter().filter(|&&t| t == image_token_id).count();
    if image_embeds.dim(0)? != n_image {
        candle::bail!(
            "{n_image} image tokens for {} image embeddings",
            image_embeds.dim(0)?
        )
    }
    let text_rows = embeds.squeeze(0)?;
    let image_embeds = image_embeds.to_dtype(text_rows.dtype())?;
    let mut rows = Vec::with_capacity(tokens.len());
    let mut next_image = 0;
    for (i, &t) in tokens.iter().enumerate() {
        if t == image_token_id {
            rows.push(image_embeds.get(next_image)?);
            next_image += 1;
        } else {
            rows.push(text_rows.get(i)?);
        }
    }
    Ok((Tensor::stack(&rows, 0)?.unsqueeze(0)?, positions, delta))
}

/// The language model with an optional vision tower.
#[derive(Debug, Clone)]
pub struct Model {
    text: qwen3_5::Model,
    vision: Option<VisionModel>,
    cfg: Config,
}

impl Model {
    /// Loads the language model, and the vision tower when `vision` is set: without it the
    /// weights of the tower are never read.
    pub fn new(cfg: &Config, vb: VarBuilder, vision: bool) -> Result<Self> {
        let text = qwen3_5::Model::new(&cfg.text, vb.clone())?;
        let vision = if vision {
            Some(VisionModel::new(&cfg.vision, vb.pp("model.visual"))?)
        } else {
            None
        };
        Ok(Self {
            text,
            vision,
            cfg: cfg.clone(),
        })
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn text(&self) -> &qwen3_5::Model {
        &self.text
    }

    pub fn text_mut(&mut self) -> &mut qwen3_5::Model {
        &mut self.text
    }

    pub fn vision(&self) -> Option<&VisionModel> {
        self.vision.as_ref()
    }

    /// Runs a `(1, seq)` prompt from cleared caches, its image tokens replaced by the embeddings
    /// of the images, and returns the `(1, seq, hidden)` final hidden states. Later steps take
    /// [`qwen3_5::Model::forward`] at the next cache position.
    pub fn forward_prompt_hidden(
        &mut self,
        ids: &Tensor,
        images: Option<Images<'_>>,
    ) -> Result<Tensor> {
        self.text.clear_kv_cache();
        let embeds = self.text.embed(ids)?;
        let Some(Images {
            pixel_values,
            grids,
        }) = images
        else {
            return self.text.forward_embeds(&embeds, 0, None);
        };
        let Some(vision) = &self.vision else {
            candle::bail!("the model was loaded without its vision tower")
        };
        let image_embeds = vision.forward(pixel_values, grids)?;
        let merge = self.cfg.vision.spatial_merge_size;
        let (embeds, positions, delta) = prompt_embeds(
            ids,
            &embeds,
            &image_embeds,
            grids,
            self.cfg.image_token_id,
            merge,
        )?;
        let hidden = self.text.forward_embeds(&embeds, 0, Some(&positions))?;
        self.text.set_rope_delta(delta);
        Ok(hidden)
    }

    /// The `(1, seq, vocab)` logits of a prompt with images; see
    /// [`Model::forward_prompt_hidden`].
    pub fn forward_prompt(&mut self, ids: &Tensor, images: Option<Images<'_>>) -> Result<Tensor> {
        let hidden = self.forward_prompt_hidden(ids, images)?;
        self.text.logits(&hidden)
    }

    /// The `(1, 1, vocab)` logits of `ids` after `offset` cached positions.
    pub fn forward(&mut self, ids: &Tensor, offset: usize) -> Result<Tensor> {
        self.text.forward(ids, offset)
    }
}
