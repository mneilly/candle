//! Qwen4-Exp text model, the architecture of Qwen3.8-Flash-Next.
//!
//! Qwen3.8-Flash-Next is a preview of the Qwen4 architecture. It keeps the Gated DeltaNet hybrid
//! of [`super::qwen3_5`] and changes four things:
//!
//! - **Gated residual.** The residual stream is widened into `hc_count` parallel streams. Each
//!   sublayer reads a gated mean of the normalized streams and writes its output back into every
//!   stream with a per-stream weight. The read of the final mixer replaces the output norm.
//! - **Qwen Sparse Attention (QSA).** On the full-attention layers, an indexer scores blocks of
//!   `indexer_compress_ratio` keys through one pooled key per block, and each query attends to
//!   its top `indexer_budget` tokens' worth of blocks plus the incomplete tail block.
//! - **N-gram embedding.** One layer adds hashed n-gram embeddings, gated per stream by the
//!   residual and passed through a dilated depthwise convolution. The table is large (51B
//!   parameters for the release), so it can stay memory-mapped on the host, see [`PleTable`].
//! - **Mixture of experts** with a gated shared expert in every layer.
//!
//! Only the text model is implemented: the vision tower and the multi-token-prediction head are
//! ignored.
//!
//! - 📝 [Tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next)
//! - 🤗 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

use crate::models::qwen3_5::{GatedDeltaNet, GatedDeltaNetConfig, RmsNorm, RotaryEmbedding};
use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use crate::utils::repeat_kv;
use candle::safetensors::MmapedSafetensors;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

pub use crate::models::qwen3_5::RopeParameters;

/// The token mixer used by a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    /// Full attention, sparse through QSA when the indexer is configured. Released configs call
    /// it `full_attention`.
    #[serde(alias = "full_attention")]
    IndexedAttention,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIds {
    /// The first id, which the n-gram hash uses as the end-of-text token.
    pub fn first(&self) -> Option<u32> {
        match self {
            Self::Single(id) => Some(*id),
            Self::Multiple(ids) => ids.first().copied(),
        }
    }
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

fn default_true() -> bool {
    true
}

fn default_hc_count() -> usize {
    4
}

fn default_hc_lowrank() -> usize {
    320
}

fn default_ngram_size() -> usize {
    3
}

fn default_heads_per_ngram() -> usize {
    8
}

fn default_ngram_vocab_size_base() -> u64 {
    20_000_000
}

fn default_ngram_vocab_divisor() -> u64 {
    128
}

fn default_seed() -> u64 {
    1234
}

/// The text config, `text_config` in the config of the multimodal checkpoints.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
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
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
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
    #[serde(default = "default_hc_count")]
    pub hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    pub hc_lowrank: usize,
    /// One-indexed ids of the layers with an n-gram embedding.
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "default_conv_kernel")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "default_ngram_vocab_divisor")]
    pub make_ngram_vocab_size_divisible_by: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub indexer_n_heads: Option<usize>,
    #[serde(default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(default)]
    pub indexer_head_dim: Option<usize>,
    #[serde(default)]
    pub indexer_budget: Option<usize>,
    #[serde(default)]
    pub indexer_compress_ratio: Option<usize>,
    /// The activation of the Gated DeltaNet output gate, `hidden_act` when unset.
    #[serde(default)]
    pub output_gate_type: Option<Activation>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIds>,
}

#[derive(serde::Deserialize)]
struct MultimodalConfig {
    text_config: Config,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
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
            if text.eos_token_id.is_none() {
                text.eos_token_id = cfg.eos_token_id
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
                            LayerType::IndexedAttention
                        } else {
                            LayerType::LinearAttention
                        }
                    })
                    .collect()
            }
        }
    }

    fn rotary_dim(&self) -> usize {
        let factor = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .or(self.partial_rotary_factor)
            .unwrap_or(1.);
        (self.head_dim as f64 * factor) as usize
    }

    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000.)
    }

    /// The one-indexed ids of the n-gram embedding layers, sorted and deduplicated as the
    /// reference does when loading the config. A layer's hash constants depend on its position
    /// in this list.
    pub fn ple_layers(&self) -> Vec<usize> {
        let mut ids = self.ple_layer_ids.clone();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn ple_embed_dim(&self) -> usize {
        self.ple_embed_dim.unwrap_or(self.hidden_size)
    }

    fn eos_token_id(&self) -> Result<u32> {
        match self.eos_token_id.as_ref().and_then(|e| e.first()) {
            Some(id) => Ok(id),
            None => candle::bail!("eos_token_id is required: the n-gram hash resets on it"),
        }
    }

    fn qsa(&self) -> Result<Option<QsaConfig>> {
        let fields = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        );
        match fields {
            (None, None, None, None, None) => Ok(None),
            (Some(n_heads), Some(kv_heads), Some(head_dim), Some(budget), Some(ratio)) => {
                if kv_heads != 1 {
                    candle::bail!("QSA requires indexer_kv_heads = 1, got {kv_heads}")
                }
                if ratio == 0 || budget % ratio != 0 {
                    candle::bail!("indexer_budget {budget} must be a multiple of indexer_compress_ratio {ratio}")
                }
                Ok(Some(QsaConfig {
                    n_heads,
                    head_dim,
                    block_topk: budget / ratio,
                    ratio,
                }))
            }
            _ => candle::bail!("the QSA config is incomplete"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct QsaConfig {
    n_heads: usize,
    head_dim: usize,
    block_topk: usize,
    ratio: usize,
}

/// A zero-centered RMSNorm applied independently to each of the `hc_count` streams.
#[derive(Debug, Clone)]
struct GroupedRmsNorm {
    /// `(groups, group_size)` weights, already offset by one.
    weight: Tensor,
    ones: Tensor,
    groups: usize,
    group_size: usize,
    eps: f32,
}

impl GroupedRmsNorm {
    fn new(groups: usize, group_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = (vb
            .get(groups * group_size, "weight")?
            .to_dtype(DType::F32)?
            .reshape((groups, group_size))?
            + 1.)?;
        let ones = Tensor::ones(group_size, DType::F32, vb.device())?;
        Ok(Self {
            weight,
            ones,
            groups,
            group_size,
            eps: eps as f32,
        })
    }

    /// Normalizes `(batch, seq, groups * group_size)` and returns `(batch, seq, groups, group_size)`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let xs4 = xs
            .to_dtype(DType::F32)?
            .reshape((b, l, self.groups, self.group_size))?
            .contiguous()?;
        candle_nn::ops::rms_norm(&xs4, &self.ones, self.eps)?
            .broadcast_mul(&self.weight)?
            .to_dtype(xs.dtype())
    }
}

/// The gated residual around a sublayer: reads a gated mean of the normalized streams, and
/// returns the per-stream weights used to write the sublayer output back.
#[derive(Debug, Clone)]
struct GatedResidual {
    hc_norm: GroupedRmsNorm,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
    hc: usize,
}

impl GatedResidual {
    fn new(cfg: &Config, use_combine: bool, vb: VarBuilder) -> Result<Self> {
        let (hc, n) = (cfg.hc_count, cfg.hidden_size);
        let inject = if use_combine {
            Some(linear_no_bias(hc * n, hc, vb.pp("block_inject_weight"))?)
        } else {
            None
        };
        Ok(Self {
            hc_norm: GroupedRmsNorm::new(hc, n, cfg.rms_norm_eps, vb.pp("hc_norm"))?,
            down: linear_no_bias(hc * n, cfg.hc_lowrank, vb.pp("input_mix_weight_down"))?,
            up: linear_no_bias(cfg.hc_lowrank, hc * n, vb.pp("input_mix_weight_up"))?,
            inject,
            hc,
        })
    }

    /// Returns the `(batch, seq, hidden)` sublayer input and the `(batch, seq, hc)` write weights.
    fn read(&self, xs: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (b, l, _) = xs.dims3()?;
        let xn4 = self.hc_norm.forward(xs)?;
        let xn = xn4.reshape((b, l, ()))?;
        let hc = self.hc as f64;
        let w = candle_nn::ops::silu(&(self.down.forward(&xn)? / hc)?)?;
        let w = candle_nn::ops::sigmoid(&self.up.forward(&w)?)?.reshape(xn4.shape())?;
        let mixed = (w * xn4)?.mean(2)?;
        let inject = match &self.inject {
            Some(inject) => Some((candle_nn::ops::sigmoid(&(inject.forward(&xn)? / hc)?)? * 2.)?),
            None => None,
        };
        Ok((mixed, inject))
    }

    /// Adds the `(batch, seq, hidden)` sublayer output to every stream of `residual`.
    fn write(residual: &Tensor, out: &Tensor, inject: &Tensor) -> Result<Tensor> {
        let (b, l, _) = residual.dims3()?;
        let update = out.unsqueeze(2)?.broadcast_mul(&inject.unsqueeze(3)?)?;
        residual + update.reshape((b, l, ()))?
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
    fn new(hidden: usize, intermediate: usize, act_fn: Activation, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
            act_fn,
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

/// Expert weights, either fused as in transformers or one MLP per expert as in some checkpoints.
#[derive(Debug, Clone)]
enum Experts {
    /// `gate_up_proj`: `(experts, 2 * intermediate, hidden)`, `down_proj`: `(experts, hidden,
    /// intermediate)`.
    Fused {
        gate_up: Tensor,
        down: Tensor,
    },
    Separate(Vec<Mlp>),
}

#[derive(Debug, Clone)]
struct SparseMoe {
    router: Tensor,
    experts: Experts,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    top_k: usize,
    norm_topk_prob: bool,
    intermediate: usize,
    act_fn: Activation,
}

impl SparseMoe {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, e, i) = (cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size);
        let vbe = vb.pp("experts");
        let experts = if vbe.contains_tensor("gate_up_proj") {
            Experts::Fused {
                gate_up: vbe.get((e, 2 * i, h), "gate_up_proj")?,
                down: vbe.get((e, h, i), "down_proj")?,
            }
        } else {
            Experts::Separate(
                (0..e)
                    .map(|j| Mlp::new(h, i, cfg.hidden_act, vbe.pp(j)))
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        let shared_i = cfg.shared_expert_intermediate_size;
        Ok(Self {
            router: vb.get((e, h), "gate.weight")?,
            experts,
            shared_expert: Mlp::new(h, shared_i, cfg.hidden_act, vb.pp("shared_expert"))?,
            shared_expert_gate: linear_no_bias(h, 1, vb.pp("shared_expert_gate"))?,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            intermediate: i,
            act_fn: cfg.hidden_act,
        })
    }

    fn expert_forward(&self, expert: usize, xs: &Tensor) -> Result<Tensor> {
        match &self.experts {
            Experts::Separate(mlps) => mlps[expert].forward(xs),
            Experts::Fused { gate_up, down } => {
                let gate_up = xs.matmul(&gate_up.get(expert)?.t()?)?;
                let gate = gate_up.narrow(D::Minus1, 0, self.intermediate)?;
                let up = gate_up.narrow(D::Minus1, self.intermediate, self.intermediate)?;
                (gate.apply(&self.act_fn)? * up)?.matmul(&down.get(expert)?.t()?)
            }
        }
    }
}

impl Module for SparseMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, h) = xs.dims3()?;
        let xs = xs.reshape((b * l, h))?;
        let logits = xs.matmul(&self.router.t()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let probs = probs.to_vec2::<f32>()?;
        let n_experts = self.router.dim(0)?;

        // Route on the host: the top-k experts of each token and their weights.
        let mut tokens = vec![vec![]; n_experts];
        let mut weights = vec![vec![]; n_experts];
        for (row, p) in probs.iter().enumerate() {
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            let top = &idx[..self.top_k];
            let sum: f32 = top.iter().map(|&e| p[e]).sum();
            for &e in top {
                tokens[e].push(row as u32);
                weights[e].push(if self.norm_topk_prob {
                    p[e] / sum
                } else {
                    p[e]
                });
            }
        }

        let mut ys = xs.zeros_like()?;
        for e in 0..n_experts {
            if tokens[e].is_empty() {
                continue;
            }
            let idx = Tensor::new(tokens[e].as_slice(), xs.device())?;
            let w = Tensor::new(weights[e].as_slice(), xs.device())?
                .to_dtype(xs.dtype())?
                .unsqueeze(1)?;
            let out = self.expert_forward(e, &xs.index_select(&idx, 0)?)?;
            ys = ys.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }
        let shared_gate = candle_nn::ops::sigmoid(&self.shared_expert_gate.forward(&xs)?)?;
        let shared = self
            .shared_expert
            .forward(&xs)?
            .broadcast_mul(&shared_gate)?;
        (ys + shared)?.reshape((b, l, h))
    }
}

/// The QSA indexer: scores blocks of `ratio` keys through their pooled key.
#[derive(Debug, Clone)]
struct QsaIndexer {
    qk_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    cfg: QsaConfig,
    /// The raw (unnormalized, unrotated) indexer key of every cached token, `(batch, seq, dim)`.
    raw_keys: Option<Tensor>,
}

impl QsaIndexer {
    fn new(cfg: QsaConfig, hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let out = (cfg.n_heads + 1) * cfg.head_dim;
        Ok(Self {
            qk_proj: linear_no_bias(hidden, out, vb.pp("index_qk_proj"))?,
            q_norm: RmsNorm::new(cfg.head_dim, eps, vb.pp("q_layernorm"))?,
            k_norm: RmsNorm::new(cfg.head_dim, eps, vb.pp("k_layernorm"))?,
            cfg,
            raw_keys: None,
        })
    }

    /// Returns the `(batch, 1, seq, kv_len)` additive mask keeping, for each query, its selected
    /// blocks and its incomplete tail block, or `None` when every query keeps all its context.
    fn mask(
        &mut self,
        xs: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
    ) -> Result<Option<Tensor>> {
        let (b, l, _) = xs.dims3()?;
        let QsaConfig {
            n_heads,
            head_dim: d,
            block_topk,
            ratio,
        } = self.cfg;
        let qk = self.qk_proj.forward(xs)?;
        let q = qk.narrow(2, 0, n_heads * d)?.reshape((b, l, n_heads, d))?;
        let keys = qk.narrow(2, n_heads * d, d)?;
        let keys = match &self.raw_keys {
            Some(prev) => Tensor::cat(&[prev, &keys], 1)?,
            None => keys.contiguous()?,
        };
        self.raw_keys = Some(keys.clone());

        let kv_len = offset + l;
        let n_blocks = kv_len / ratio;
        if n_blocks <= block_topk {
            // Every query has at most block_topk complete blocks, and keeps them all.
            return Ok(None);
        }
        let dev = xs.device();
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let q = rope.apply(&q, offset)?;

        // Pooled keys: the mean of each complete block, normed and rotated to its first token.
        let pooled = keys
            .narrow(1, 0, n_blocks * ratio)?
            .to_dtype(DType::F32)?
            .reshape((b, n_blocks, ratio, d))?
            .mean(2)?
            .to_dtype(keys.dtype())?;
        let pooled = self.k_norm.forward(&pooled)?;
        let starts: Vec<u32> = (0..n_blocks).map(|j| (j * ratio) as u32).collect();
        let starts = Tensor::new(starts.as_slice(), dev)?;
        let pooled = rope.apply_at(&pooled.unsqueeze(1)?, &starts)?.squeeze(1)?;

        // scores[b, t, j] = sum over heads of relu(q . k_j) / sqrt(d).
        let q = q
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .reshape((b, l * n_heads, d))?;
        let k_t = pooled.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let scores = (q
            .contiguous()?
            .matmul(&k_t)?
            .relu()?
            .reshape((b, l, n_heads, n_blocks))?
            .sum(2)?
            / (d as f64).sqrt())?;
        let scores = scores.to_vec3::<f32>()?;

        let mut mask = vec![f32::NEG_INFINITY; b * l * kv_len];
        for (bi, batch_scores) in scores.iter().enumerate() {
            for (t, row) in batch_scores.iter().enumerate() {
                let base = (bi * l + t) * kv_len;
                qsa_keep(
                    row,
                    offset + t,
                    ratio,
                    block_topk,
                    &mut mask[base..base + kv_len],
                );
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, l, kv_len), dev)?;
        Ok(Some(mask))
    }

    fn clear_cache(&mut self) {
        self.raw_keys = None
    }
}

/// Zeroes the entries of `mask`, the additive mask row of the query at position `pos`, for the
/// keys it attends to under QSA: the tokens of its `block_topk` highest-scoring complete blocks of
/// `ratio` tokens (ties going to the earlier block), and its incomplete tail block. `scores` holds
/// one score per complete block of the whole key sequence.
fn qsa_keep(scores: &[f32], pos: usize, ratio: usize, block_topk: usize, mask: &mut [f32]) {
    let complete = (pos + 1) / ratio;
    let mut blocks: Vec<usize> = (0..complete).collect();
    if complete > block_topk {
        // Best score first, ties to the earlier block.
        let order = |&x: &usize, &y: &usize| scores[y].total_cmp(&scores[x]).then(x.cmp(&y));
        #[cfg(not(kani))]
        blocks.sort_by(order);
        // std's sort is too large for CBMC, so proofs rank with an insertion sort under the
        // same order.
        #[cfg(kani)]
        for i in 1..blocks.len() {
            let mut j = i;
            while j > 0 && order(&blocks[j - 1], &blocks[j]).is_gt() {
                blocks.swap(j - 1, j);
                j -= 1;
            }
        }
        blocks.truncate(block_topk);
    }
    for j in blocks {
        mask[j * ratio..(j + 1) * ratio].fill(0.);
    }
    mask[complete * ratio..pos + 1].fill(0.);
}

/// Full attention with an output gate, QK-norm, partial RoPE and optional QSA sparsity.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    indexer: Option<QsaIndexer>,
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
        let indexer = match cfg.qsa()? {
            Some(qsa) => Some(QsaIndexer::new(qsa, h, cfg.rms_norm_eps, vb.pp("indexer"))?),
            None => None,
        };
        Ok(Self {
            q_proj: linear_b(h, nh * hd * 2, bias, vb.pp("q_proj"))?,
            k_proj: linear_b(h, nkv * hd, bias, vb.pp("k_proj"))?,
            v_proj: linear_b(h, nkv * hd, bias, vb.pp("v_proj"))?,
            o_proj: linear_b(nh * hd, h, bias, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            indexer,
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
        let qsa_mask = match &mut self.indexer {
            Some(indexer) => indexer.mask(xs, &self.rotary_emb, offset)?,
            None => None,
        };

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
        let mut scores = (q.contiguous()?.matmul(&k.t()?)? * scale)?.to_dtype(DType::F32)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.to_dtype(DType::F32)?)?;
        }
        if let Some(m) = qsa_mask {
            scores = scores.broadcast_add(&m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ys = probs.to_dtype(v.dtype())?.matmul(&v)?;
        let ys = ys.transpose(1, 2)?.reshape((b, l, self.num_heads * hd))?;
        let ys = (ys * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&ys)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
        if let Some(indexer) = &mut self.indexer {
            indexer.clear_cache()
        }
    }
}

const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const PRIME_1: u64 = 10007;

fn splitmix64(value: u64) -> u64 {
    let v = value.wrapping_add(SPLITMIX_GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
    v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2
    }
    true
}

/// The hash constants of one n-gram embedding layer, as derived by the reference implementation.
#[derive(Debug, Clone, PartialEq)]
struct NgramHash {
    multipliers: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    /// The table row count, the vocabulary sizes summed and padded.
    rows: u64,
}

impl NgramHash {
    fn new(cfg: &Config, ple_layer_index: usize) -> Self {
        let n_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let max_long = (1u64 << 63) - 1;
        let half_bound = (max_long / cfg.vocab_size.max(1) as u64 / 2).max(1);
        let base_seed = cfg
            .seed
            .wrapping_add(PRIME_1.wrapping_mul(ple_layer_index as u64));
        let multipliers = (0..cfg.ngram_size as u64)
            .map(|i| {
                let v = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i + 1));
                2 * (splitmix64(v) % half_bound) + 1
            })
            .collect();

        // Head h uses the (global_h + 1)-th prime from ngram_vocab_size_base on.
        let mut head_vocab_sizes = Vec::with_capacity(n_heads);
        let mut head_offsets = Vec::with_capacity(n_heads);
        let first_head = ple_layer_index * n_heads;
        let mut prime = cfg.ngram_vocab_size_base - 1;
        let mut found = 0;
        let mut total = 0;
        while head_vocab_sizes.len() < n_heads {
            prime += 1;
            while !is_prime(prime) {
                prime += 1
            }
            found += 1;
            if found > first_head {
                head_vocab_sizes.push(prime);
                head_offsets.push(total);
                total += prime;
            }
        }
        let div = cfg.make_ngram_vocab_size_divisible_by;
        Self {
            multipliers,
            head_vocab_sizes,
            head_offsets,
            rows: total.div_ceil(div) * div,
        }
    }

    /// The table rows of the n-gram heads for each position of `ids`, given the previous
    /// `ngram_size - 1` tokens. The window resets after each end-of-text token.
    fn rows(&self, context: &[u32], ids: &[u32], heads_per_ngram: usize, eos: u32) -> Vec<u32> {
        let ngram_size = self.multipliers.len();
        let hist: Vec<u64> = context
            .iter()
            .chain(ids.iter())
            .map(|&t| t as u64)
            .collect();
        // start[p]: where the segment holding p starts, one past the last earlier EOS.
        let mut start = vec![0usize; hist.len()];
        let mut seg = 0;
        for (p, &t) in hist.iter().enumerate() {
            start[p] = seg;
            if t == eos as u64 {
                seg = p + 1
            }
        }
        let shifted = |p: usize, shift: usize| -> u64 {
            if p - start[p] >= shift {
                hist[p - shift]
            } else {
                eos as u64
            }
        };
        let mut rows = Vec::with_capacity(ids.len() * (ngram_size - 1) * heads_per_ngram);
        for p in context.len()..hist.len() {
            for n in 2..=ngram_size {
                let mut mixed = shifted(p, 0).wrapping_mul(self.multipliers[0]);
                for pos in 1..n {
                    mixed ^= shifted(p, pos).wrapping_mul(self.multipliers[pos]);
                }
                let first = (n - 2) * heads_per_ngram;
                for h in first..first + heads_per_ngram {
                    let row = mixed % self.head_vocab_sizes[h] + self.head_offsets[h];
                    rows.push(row as u32);
                }
            }
        }
        rows
    }
}

/// The n-gram embedding table, `(rows, head_dim)`.
#[derive(Clone)]
pub enum PleTable {
    /// The whole table as one tensor.
    Tensor(Tensor),
    /// The table left in memory-mapped safetensors files, possibly split into row shards; only
    /// the rows a forward pass needs are read.
    Mmap {
        st: Arc<MmapedSafetensors>,
        /// Each shard's tensor name and first row, in row order.
        shards: Vec<(String, usize)>,
        head_dim: usize,
    },
}

impl std::fmt::Debug for PleTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(t) => f.debug_tuple("Tensor").field(t).finish(),
            Self::Mmap {
                shards, head_dim, ..
            } => f
                .debug_struct("Mmap")
                .field("shards", &shards.len())
                .field("head_dim", head_dim)
                .finish(),
        }
    }
}

impl PleTable {
    /// Finds the table of layer `layer` (zero-based) in memory-mapped checkpoint files, under
    /// `prefix` (e.g. `model.language_model`).
    pub fn from_mmaped(st: Arc<MmapedSafetensors>, prefix: &str, layer: usize) -> Result<Self> {
        let base = format!("{prefix}.layers.{layer}.ple.ple_embedding.ngram_embedding");
        let names: Vec<String> = if st.get(&format!("{base}.weight")).is_ok() {
            vec![format!("{base}.weight")]
        } else {
            let mut names = vec![];
            while st
                .get(&format!("{base}.shard_{}.weight", names.len()))
                .is_ok()
            {
                names.push(format!("{base}.shard_{}.weight", names.len()))
            }
            names
        };
        if names.is_empty() {
            candle::bail!("no n-gram embedding table under {base}")
        }
        let mut shards = Vec::with_capacity(names.len());
        let (mut row, mut head_dim) = (0, 0);
        for name in names {
            let view = st.get(&name)?;
            let shape = view.shape();
            head_dim = shape[1];
            shards.push((name, row));
            row += shape[0];
        }
        Ok(Self::Mmap {
            st,
            shards,
            head_dim,
        })
    }

    fn from_vb(vb: &VarBuilder, head_dim: usize, rows: usize) -> Result<Self> {
        if vb.contains_tensor("weight") {
            return Ok(Self::Tensor(vb.get((rows, head_dim), "weight")?));
        }
        let mut shards = vec![];
        while vb.contains_tensor(&format!("shard_{}.weight", shards.len())) {
            let name = format!("shard_{}.weight", shards.len());
            shards.push(vb.get_unchecked(&name)?);
        }
        if shards.is_empty() {
            candle::bail!("no n-gram embedding table")
        }
        Ok(Self::Tensor(Tensor::cat(&shards, 0)?))
    }

    /// Gathers the given rows as a `(rows.len(), head_dim)` tensor on `device`.
    fn gather(&self, rows: &[u32], device: &Device, dtype: DType) -> Result<Tensor> {
        match self {
            Self::Tensor(t) => {
                let idx = Tensor::new(rows, t.device())?;
                t.index_select(&idx, 0)?.to_device(device)?.to_dtype(dtype)
            }
            Self::Mmap {
                st,
                shards,
                head_dim,
            } => {
                let first = st.get(&shards[0].0)?;
                let st_dtype: DType = first.dtype().try_into()?;
                let row_bytes = head_dim * st_dtype.size_in_bytes();
                let mut buf = Vec::with_capacity(rows.len() * row_bytes);
                for &row in rows {
                    let row = row as usize;
                    let s = shards.partition_point(|(_, start)| *start <= row) - 1;
                    let (name, start) = &shards[s];
                    let view = st.get(name)?;
                    let offset = (row - start) * row_bytes;
                    buf.extend_from_slice(&view.data()[offset..offset + row_bytes]);
                }
                Tensor::from_raw_buffer(&buf, st_dtype, &[rows.len(), *head_dim], &Device::Cpu)?
                    .to_device(device)?
                    .to_dtype(dtype)
            }
        }
    }
}

/// The n-gram embedding layer: hashed n-gram embeddings gated into every residual stream.
#[derive(Debug, Clone)]
struct Ple {
    table: PleTable,
    hash: NgramHash,
    key_proj: Linear,
    value_proj: Linear,
    norm_key: GroupedRmsNorm,
    norm_query: GroupedRmsNorm,
    norm_conv: GroupedRmsNorm,
    /// The dilated depthwise convolution weight, `(hc * hidden, kernel)`.
    conv_weight: Tensor,
    dilation: usize,
    heads_per_ngram: usize,
    eos: u32,
    hidden: usize,
    /// The previous `ngram_size - 1` tokens of each sequence.
    token_context: Option<Vec<Vec<u32>>>,
    /// The last `(kernel - 1) * dilation` convolution inputs, `(batch, channels, len)`.
    conv_state: Option<Tensor>,
}

impl Ple {
    fn new(
        cfg: &Config,
        ple_layer_index: usize,
        table: Option<PleTable>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (hc, n) = (cfg.hc_count, cfg.hidden_size);
        let ple_dim = cfg.ple_embed_dim();
        let n_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        if !ple_dim.is_multiple_of(n_heads) {
            candle::bail!("ple_embed_dim {ple_dim} is not a multiple of the {n_heads} n-gram heads")
        }
        let hash = NgramHash::new(cfg, ple_layer_index);
        let table = match table {
            Some(table) => table,
            None => PleTable::from_vb(
                &vb.pp("ple_embedding.ngram_embedding"),
                ple_dim / n_heads,
                hash.rows as usize,
            )?,
        };
        let eps = cfg.rms_norm_eps;
        let kernel = cfg.ple_conv_kernel_size;
        let conv_weight = vb
            .get((hc * n, 1, kernel), "conv1d.weight")?
            .reshape((hc * n, kernel))?;
        Ok(Self {
            table,
            hash,
            key_proj: linear_no_bias(ple_dim, hc * n, vb.pp("key_proj"))?,
            value_proj: linear_no_bias(ple_dim, n, vb.pp("value_proj"))?,
            norm_key: GroupedRmsNorm::new(hc, n, eps, vb.pp("norm_key"))?,
            norm_query: GroupedRmsNorm::new(hc, n, eps, vb.pp("norm_query"))?,
            norm_conv: GroupedRmsNorm::new(hc, n, eps, vb.pp("norm_conv"))?,
            conv_weight,
            dilation: cfg.ngram_size,
            heads_per_ngram: cfg.heads_per_ngram,
            eos: cfg.eos_token_id()?,
            hidden: n,
            token_context: None,
            conv_state: None,
        })
    }

    fn embed(&mut self, ids: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let (b, l) = ids.dims2()?;
        let ids = ids.to_vec2::<u32>()?;
        let ctx_len = self.hash.multipliers.len() - 1;
        let contexts = self
            .token_context
            .take()
            .unwrap_or_else(|| vec![vec![self.eos; ctx_len]; b]);
        let mut rows = Vec::with_capacity(b * l * self.hash.head_vocab_sizes.len());
        let mut new_contexts = Vec::with_capacity(b);
        for (ctx, ids) in contexts.iter().zip(ids.iter()) {
            rows.extend(self.hash.rows(ctx, ids, self.heads_per_ngram, self.eos));
            let hist: Vec<u32> = ctx.iter().chain(ids.iter()).copied().collect();
            new_contexts.push(hist[hist.len() - ctx_len..].to_vec());
        }
        self.token_context = Some(new_contexts);
        self.table.gather(&rows, device, dtype)?.reshape((b, l, ()))
    }

    /// The dilated causal depthwise convolution followed by SiLU, continuing from the cached
    /// inputs.
    fn conv(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        let kernel = self.conv_weight.dim(1)?;
        let state_len = (kernel - 1) * self.dilation;
        let xs = xs.transpose(1, 2)?.to_dtype(self.conv_weight.dtype())?;
        let state = match &self.conv_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((b, c, state_len), xs.dtype(), xs.device())?,
        };
        let full = Tensor::cat(&[&state, &xs], 2)?;
        let mut ys = full
            .narrow(2, 0, l)?
            .broadcast_mul(&self.conv_weight.narrow(1, 0, 1)?)?;
        for j in 1..kernel {
            let w = self.conv_weight.narrow(1, j, 1)?;
            ys = (ys + full.narrow(2, j * self.dilation, l)?.broadcast_mul(&w)?)?;
        }
        self.conv_state = Some(full.narrow(2, l, state_len)?.contiguous()?);
        candle_nn::ops::silu(&ys)?.transpose(1, 2)
    }

    /// Returns the `(batch, seq, hc * hidden)` update of the residual streams.
    fn forward(&mut self, xs: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let emb = self.embed(ids, xs.device(), xs.dtype())?;
        let key = self.norm_key.forward(&self.key_proj.forward(&emb)?)?;
        let value = self.value_proj.forward(&emb)?;
        let query = self.norm_query.forward(xs)?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (self.hidden as f64).sqrt())?;
        // sign(g) * sqrt(max(|g|, 1e-6))
        let gate = (gate.abs()?.maximum(1e-6)?.sqrt()? * gate.sign()?)?;
        let gated = candle_nn::ops::sigmoid(&gate)?.broadcast_mul(&value.unsqueeze(2)?)?;
        let gated = gated.reshape((b, l, ()))?;
        let normed = self.norm_conv.forward(&gated)?.reshape((b, l, ()))?;
        let conv = self.conv(&normed)?.to_dtype(gated.dtype())?;
        gated + conv
    }

    fn clear_cache(&mut self) {
        self.token_context = None;
        self.conv_state = None;
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    Linear(GatedDeltaNet),
    Indexed(Attention),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    ple: Option<Ple>,
    attn_hyper_connection: GatedResidual,
    mixer: TokenMixer,
    mlp_hyper_connection: GatedResidual,
    mlp: SparseMoe,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        layer_type: LayerType,
        ple: Option<(usize, Option<PleTable>)>,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mixer = match layer_type {
            LayerType::LinearAttention => {
                let gdn = GatedDeltaNetConfig {
                    hidden_size: cfg.hidden_size,
                    num_k_heads: cfg.linear_num_key_heads,
                    num_v_heads: cfg.linear_num_value_heads,
                    head_k_dim: cfg.linear_key_head_dim,
                    head_v_dim: cfg.linear_value_head_dim,
                    conv_kernel: cfg.linear_conv_kernel_dim,
                    rms_norm_eps: cfg.rms_norm_eps,
                    gate_act: cfg.output_gate_type.unwrap_or(cfg.hidden_act),
                };
                TokenMixer::Linear(GatedDeltaNet::new(gdn, vb.pp("linear_attn"))?)
            }
            LayerType::IndexedAttention => {
                TokenMixer::Indexed(Attention::new(cfg, rotary, vb.pp("self_attn"))?)
            }
        };
        let ple = match ple {
            Some((index, table)) => Some(Ple::new(cfg, index, table, vb.pp("ple"))?),
            None => None,
        };
        Ok(Self {
            ple,
            attn_hyper_connection: GatedResidual::new(cfg, true, vb.pp("attn_hyper_connection"))?,
            mixer,
            mlp_hyper_connection: GatedResidual::new(cfg, true, vb.pp("mlp_hyper_connection"))?,
            mlp: SparseMoe::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        ids: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        if let Some(ple) = &mut self.ple {
            xs = (&xs + ple.forward(&xs, ids)?)?;
        }
        let (h, inject) = self.attn_hyper_connection.read(&xs)?;
        let h = match &mut self.mixer {
            TokenMixer::Linear(m) => m.forward(&h)?,
            TokenMixer::Indexed(m) => m.forward(&h, mask, offset)?,
        };
        let xs = GatedResidual::write(&xs, &h, &inject.unwrap())?;
        let (h, inject) = self.mlp_hyper_connection.read(&xs)?;
        let h = self.mlp.forward(&h)?;
        GatedResidual::write(&xs, &h, &inject.unwrap())
    }

    fn clear_cache(&mut self) {
        match &mut self.mixer {
            TokenMixer::Linear(m) => m.clear_cache(),
            TokenMixer::Indexed(m) => m.clear_cache(),
        }
        if let Some(ple) = &mut self.ple {
            ple.clear_cache()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    hyper_connection_mixer: GatedResidual,
    lm_head: Linear,
    hc_count: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    /// The prefix of the text model weights: `model.language_model` in the multimodal
    /// checkpoints, `model` in text-only ones.
    pub fn weight_prefix(vb: &VarBuilder) -> &'static str {
        if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        }
    }

    /// Loads the model, reading the n-gram embedding tables through `vb` like other weights.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_ple_tables(cfg, vb, vec![])
    }

    /// Loads the model with the given n-gram embedding tables, one per layer of
    /// [`Config::ple_layers`] and in that order. Layers without a table read theirs through `vb`.
    pub fn new_with_ple_tables(
        cfg: &Config,
        vb: VarBuilder,
        tables: Vec<PleTable>,
    ) -> Result<Self> {
        let vb_m = vb.pp(Self::weight_prefix(&vb));
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new_partial(
            vb.dtype(),
            cfg.rotary_dim(),
            cfg.rope_theta(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);
        let layer_types = cfg.layer_types();
        if layer_types.len() != cfg.num_hidden_layers {
            candle::bail!(
                "{} layer types for {} layers",
                layer_types.len(),
                cfg.num_hidden_layers
            )
        }
        let ple_ids = cfg.ple_layers();
        let mut tables: Vec<Option<PleTable>> = tables.into_iter().map(Some).collect();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, &t) in layer_types.iter().enumerate() {
            let ple = ple_ids
                .iter()
                .position(|&id| id == i + 1)
                .map(|index| (index, tables.get_mut(index).and_then(Option::take)));
            let vb_l = vb_m.pp("layers").pp(i);
            layers.push(DecoderLayer::new(cfg, t, ple, rotary.clone(), vb_l)?);
        }
        let hyper_connection_mixer =
            GatedResidual::new(cfg, false, vb_m.pp("hyper_connection_mixer"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            hyper_connection_mixer,
            lm_head,
            hc_count: cfg.hc_count,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Runs `input` `(batch, seq)` after `offset` already-processed tokens and returns the
    /// `(batch, seq, hidden)` final hidden states.
    pub fn forward_hidden(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let mask = if l > 1 {
            Some(crate::utils::build_additive_causal_mask(
                l,
                offset,
                None,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };
        let emb = self.embed_tokens.forward(input)?;
        // The residual starts as hc_count copies of the embedding.
        let mut xs = Tensor::cat(&vec![&emb; self.hc_count], D::Minus1)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, input, mask.as_ref(), offset)?;
        }
        Ok(self.hyper_connection_mixer.read(&xs)?.0)
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

    /// Clears the caches and recurrent states, to start a new sequence.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache()
        }
    }
}

/// The n-gram hash constants of the `ple_layer_index`-th n-gram embedding layer: the per-position
/// multipliers, the vocabulary size of each head and each head's first table row.
pub fn ngram_hash_constants(
    cfg: &Config,
    ple_layer_index: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let h = NgramHash::new(cfg, ple_layer_index);
    (h.multipliers, h.head_vocab_sizes, h.head_offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes() {
        let primes: Vec<u64> = (0..30).filter(|&v| is_prime(v)).collect();
        assert_eq!(primes, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn ngram_rows_reset_on_eos() {
        let hash = NgramHash {
            multipliers: vec![3, 5, 7],
            head_vocab_sizes: vec![1000, 1000],
            head_offsets: vec![0, 1000],
            rows: 2048,
        };
        let eos = 0;
        // One head per n-gram order: bigrams then trigrams.
        let rows = hash.rows(&[eos, eos], &[4, 9, eos, 6], 1, eos);
        let bigram = |a: u64, b: u64| (a * 3) ^ (b * 5);
        let trigram = |a: u64, b: u64, c: u64| (a * 3) ^ (b * 5) ^ (c * 7);
        let expected = [
            bigram(4, 0) % 1000,
            trigram(4, 0, 0) % 1000 + 1000,
            bigram(9, 4) % 1000,
            trigram(9, 4, 0) % 1000 + 1000,
            bigram(0, 9) % 1000,
            trigram(0, 9, 4) % 1000 + 1000,
            // After the EOS the window restarts.
            bigram(6, 0) % 1000,
            trigram(6, 0, 0) % 1000 + 1000,
        ];
        let rows: Vec<u64> = rows.iter().map(|&r| r as u64).collect();
        assert_eq!(rows, expected);
    }
}

/// Bounded proofs of the QSA selection and the n-gram hashing, run with
/// `cargo kani -p candle-transformers`.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// QSA never lets a query see a future token, always keeps the tail block, keeps whole
    /// blocks, and keeps exactly the `block_topk` best-scoring complete blocks.
    #[kani::proof]
    #[kani::unwind(8)]
    fn qsa_keeps_the_top_past_blocks_and_the_tail() {
        const KV: usize = 6;
        let ratio: usize = kani::any();
        let block_topk: usize = kani::any();
        let pos: usize = kani::any();
        kani::assume((1..=3).contains(&ratio) && block_topk <= 2 && pos < KV);
        let scores: [f32; KV] = kani::any();
        let mut mask = [f32::NEG_INFINITY; KV];
        qsa_keep(&scores, pos, ratio, block_topk, &mut mask);

        let kept = |j: usize| mask[j] == 0.;
        let complete = (pos + 1) / ratio;
        for j in 0..KV {
            if kept(j) {
                assert!(j <= pos, "a future token is visible");
            }
        }
        for j in complete * ratio..=pos {
            assert!(kept(j), "the tail block is dropped");
        }
        let mut n_kept = 0;
        for b in 0..complete {
            for j in b * ratio..(b + 1) * ratio {
                assert_eq!(kept(j), kept(b * ratio), "a block is split");
            }
            if kept(b * ratio) {
                n_kept += 1;
                for c in 0..complete {
                    if !kept(c * ratio) {
                        // A kept block scores at least as high as a dropped one, and wins ties
                        // only by coming first.
                        let ord = scores[b].total_cmp(&scores[c]);
                        assert!(ord.is_gt() || (ord.is_eq() && b < c));
                    }
                }
            }
        }
        assert_eq!(n_kept, complete.min(block_topk));
    }

    fn tiny_hash() -> NgramHash {
        // Odd multipliers below 2^63 / 50 as the reference derives for a 50-token vocabulary,
        // two heads per n-gram order, and the first primes from 200 as head vocabularies.
        NgramHash {
            multipliers: vec![3612826517417929, 34937402946437465, 61226416566306291],
            head_vocab_sizes: vec![211, 223, 227, 229],
            head_offsets: vec![0, 211, 434, 661],
            rows: 896,
        }
    }

    /// Every row a token gathers lies in the range of its head.
    #[kani::proof]
    #[kani::unwind(6)]
    fn ngram_rows_stay_in_their_head() {
        let hash = tiny_hash();
        let context: [u32; 2] = [kani::any(), kani::any()];
        let ids: [u32; 2] = [kani::any(), kani::any()];
        kani::assume(context.iter().chain(ids.iter()).all(|&t| t < 50));
        let rows = hash.rows(&context, &ids, 2, 3);
        assert_eq!(rows.len(), 2 * 4);
        for (k, &row) in rows.iter().enumerate() {
            let h = k % 4;
            let row = row as u64;
            assert!(row >= hash.head_offsets[h]);
            assert!(row < hash.head_offsets[h] + hash.head_vocab_sizes[h]);
            assert!(row < hash.rows);
        }
    }

    /// After an end-of-text token, the rows only depend on the tokens that follow it.
    #[kani::proof]
    #[kani::unwind(6)]
    fn ngram_window_resets_after_eos() {
        let hash = tiny_hash();
        let eos = 3;
        let (a, b): (u32, u32) = (kani::any(), kani::any());
        let (x, y): (u32, u32) = (kani::any(), kani::any());
        kani::assume([a, b, x, y].iter().all(|&t| t < 50));
        let rows_1 = hash.rows(&[a, b], &[eos, x, y], 2, eos);
        let rows_2 = hash.rows(&[b, a], &[eos, x, y], 2, eos);
        // Positions 1 and 2 follow the EOS at position 0.
        assert_eq!(rows_1[4..], rows_2[4..]);
    }

    /// `is_prime` agrees with trial division by every smaller number.
    #[kani::proof]
    #[kani::unwind(130)]
    fn is_prime_is_trial_division() {
        let v: u64 = kani::any();
        kani::assume(v < 128);
        let naive = v >= 2 && (2..v).all(|d| v % d != 0);
        assert_eq!(is_prime(v), naive);
    }
}
