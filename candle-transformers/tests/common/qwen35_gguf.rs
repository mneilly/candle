//! Writes `qwen35` GGUF files from the Qwen3.5 fixtures as llama.cpp's converter does
//! (PrismML-Eng/llama.cpp, `conversion/qwen.py`), for the GGUF tests.

use candle::quantized::gguf_file::{self, Value};
use candle::quantized::{GgmlDType, QTensor};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_5::{Config, L2Norm, Model};
use candle_transformers::models::qwen3_5;
use std::collections::HashMap;
use std::io::Cursor;

use super::{load, read};

/// A Hadamard fold: every 2D weight except the GDN projections to the value heads, the token
/// table rotated for lookups, and with `tied` the output head bound to it (version 2).
#[derive(Clone, Copy)]
pub struct Fold {
    pub block: usize,
    pub signs: bool,
    pub tied: bool,
}

#[derive(Clone, Copy, Default)]
pub struct Opts {
    pub fold: Option<Fold>,
    pub fused_qkv: bool,
    /// Halve `ffn_down` and `attn_output`, with a scale of 2.
    pub scales: bool,
    /// A draft head over every other token.
    pub draft_head: bool,
    /// MTP layers with their own copies of the token table and LM head.
    pub mtp_own_head: bool,
    pub dtype: Option<GgmlDType>,
}

pub struct Hf {
    pub t: HashMap<String, Tensor>,
    pub cfg: qwen3_5::Config,
    /// The prefix of the trunk weights, `model.` or `model.language_model.`.
    pub prefix: String,
}

impl Hf {
    pub fn fixture(variant: &str) -> Self {
        let cfg =
            qwen3_5::Config::from_json(&read(&format!("qwen3_5/{variant}/config.json"))).unwrap();
        let t = load(&format!("qwen3_5/{variant}/model.safetensors")).unwrap();
        let prefix = if t.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        Self {
            t,
            cfg,
            prefix: prefix.to_string(),
        }
    }

    pub fn get(&self, name: &str) -> Tensor {
        match self.t.get(&format!("{}{name}", self.prefix)) {
            Some(t) => t.clone(),
            None => self.t[name].clone(),
        }
    }
}

/// Grouped value heads `g * rep + r` to tiled `r * n_k + g` along `dim`.
pub fn tile(t: &Tensor, dim: usize, nk: usize, rep: usize, hd: usize) -> Tensor {
    let mut shape = t.dims().to_vec();
    let mut split = shape.clone();
    split.splice(dim..dim + 1, [nk, rep, hd]);
    let t = t.reshape(split).unwrap().transpose(dim, dim + 1).unwrap();
    shape[dim] = nk * rep * hd;
    t.contiguous().unwrap().reshape(shape).unwrap()
}

/// `H[i, j] = (-1)^popcount(i & j) / sqrt(n)`.
pub fn hadamard(n: usize) -> Tensor {
    let s = 1. / (n as f32).sqrt();
    let h: Vec<f32> = (0..n * n)
        .map(|k| {
            if ((k / n) & (k % n)).count_ones() % 2 == 1 {
                -s
            } else {
                s
            }
        })
        .collect();
    Tensor::from_vec(h, (n, n), &Device::Cpu).unwrap()
}

/// Signs from a fixed pattern, one per input feature.
pub fn signs(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| if (i * 7 + i / 3) % 5 < 2 { -1 } else { 1 })
        .collect()
}

/// `W D H`: the signs then the block rotation on the input dimension.
pub fn fold_weight(w: &Tensor, f: Fold) -> Tensor {
    let (o, i) = w.dims2().unwrap();
    let mut w = w.clone();
    if f.signs {
        let s: Vec<f32> = signs(i).iter().map(|&s| s as f32).collect();
        w = w
            .broadcast_mul(&Tensor::new(s, &Device::Cpu).unwrap())
            .unwrap();
    }
    w.reshape((o * i / f.block, f.block))
        .unwrap()
        .matmul(&hadamard(f.block))
        .unwrap()
        .reshape((o, i))
        .unwrap()
}

pub fn u32v(v: usize) -> Value {
    Value::U32(v as u32)
}

/// Writes the GGUF file of a Qwen3.5 checkpoint.
pub fn convert(hf: &Hf, opts: Opts) -> Vec<u8> {
    let cfg = &hf.cfg;
    let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
    let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
    let rep = nv / nk;
    let n_mtp = if hf.t.contains_key("mtp.fc.weight") {
        cfg.mtp_num_hidden_layers
    } else {
        0
    };
    let mut ts: Vec<(String, Tensor)> = vec![];
    let plus1 = |t: Tensor| (t + 1.).unwrap();
    let layer = |ts: &mut Vec<(String, Tensor)>, hp: &str, gp: &str, full: bool| {
        let g = |n: &str| hf.get(&format!("{hp}.{n}"));
        let mut put = |n: &str, t: Tensor| ts.push((format!("{gp}.{n}"), t));
        put("attn_norm.weight", plus1(g("input_layernorm.weight")));
        put(
            "post_attention_norm.weight",
            plus1(g("post_attention_layernorm.weight")),
        );
        for (h, n) in [
            ("gate_proj", "ffn_gate"),
            ("up_proj", "ffn_up"),
            ("down_proj", "ffn_down"),
        ] {
            put(&format!("{n}.weight"), g(&format!("mlp.{h}.weight")));
        }
        if full {
            let (q, k, v) = (
                g("self_attn.q_proj.weight"),
                g("self_attn.k_proj.weight"),
                g("self_attn.v_proj.weight"),
            );
            if opts.fused_qkv {
                put("attn_qkv.weight", Tensor::cat(&[q, k, v], 0).unwrap());
            } else {
                put("attn_q.weight", q);
                put("attn_k.weight", k);
                put("attn_v.weight", v);
            }
            put("attn_output.weight", g("self_attn.o_proj.weight"));
            put("attn_q_norm.weight", plus1(g("self_attn.q_norm.weight")));
            put("attn_k_norm.weight", plus1(g("self_attn.k_norm.weight")));
        } else {
            let qkv = g("linear_attn.in_proj_qkv.weight");
            let v = tile(
                &qkv.narrow(0, 2 * nk * dk, nv * dv).unwrap(),
                0,
                nk,
                rep,
                dv,
            );
            let qkv = Tensor::cat(&[qkv.narrow(0, 0, 2 * nk * dk).unwrap(), v], 0).unwrap();
            put("attn_qkv.weight", qkv);
            put(
                "attn_gate.weight",
                tile(&g("linear_attn.in_proj_z.weight"), 0, nk, rep, dv),
            );
            put(
                "ssm_beta.weight",
                tile(&g("linear_attn.in_proj_b.weight"), 0, nk, rep, 1),
            );
            put(
                "ssm_alpha.weight",
                tile(&g("linear_attn.in_proj_a.weight"), 0, nk, rep, 1),
            );
            let a = g("linear_attn.A_log").exp().unwrap().neg().unwrap();
            put(
                "ssm_a",
                tile(&a.unsqueeze(1).unwrap(), 0, nk, rep, 1)
                    .flatten_all()
                    .unwrap(),
            );
            let dt = g("linear_attn.dt_bias").unsqueeze(1).unwrap();
            put(
                "ssm_dt.bias",
                tile(&dt, 0, nk, rep, 1).flatten_all().unwrap(),
            );
            let conv = g("linear_attn.conv1d.weight").squeeze(1).unwrap();
            let cv = tile(
                &conv.narrow(0, 2 * nk * dk, nv * dv).unwrap(),
                0,
                nk,
                rep,
                dv,
            );
            let conv = Tensor::cat(&[conv.narrow(0, 0, 2 * nk * dk).unwrap(), cv], 0).unwrap();
            put("ssm_conv1d.weight", conv);
            put("ssm_norm.weight", g("linear_attn.norm.weight"));
            let out = g("linear_attn.out_proj.weight");
            // A folded output projection keeps the grouped order, the runtime permutes instead.
            let out = if opts.fold.is_some() {
                out
            } else {
                tile(&out, 1, nk, rep, dv)
            };
            put("ssm_out.weight", out);
        }
    };
    let types = cfg.layer_types();
    for (i, t) in types.iter().enumerate() {
        let full = *t == qwen3_5::LayerType::FullAttention;
        layer(&mut ts, &format!("layers.{i}"), &format!("blk.{i}"), full);
    }
    let n = cfg.num_hidden_layers;
    for i in 0..n_mtp {
        let gp = format!("blk.{}", n + i);
        layer(&mut ts, &format!("mtp.layers.{i}"), &gp, true);
        ts.push((
            format!("{gp}.nextn.eh_proj.weight"),
            hf.t["mtp.fc.weight"].clone(),
        ));
        for (h, g) in [
            ("pre_fc_norm_embedding", "enorm"),
            ("pre_fc_norm_hidden", "hnorm"),
            ("norm", "shared_head_norm"),
        ] {
            let w = plus1(hf.t[&format!("mtp.{h}.weight")].clone());
            ts.push((format!("{gp}.nextn.{g}.weight"), w));
        }
        if opts.mtp_own_head {
            let n = |s: &str| format!("{gp}.nextn.{s}.weight");
            ts.push((n("embed_tokens"), hf.get("embed_tokens.weight")));
            ts.push((n("shared_head_head"), hf.t["lm_head.weight"].clone()));
        }
        if opts.draft_head && i == 0 {
            let lm = hf.t["lm_head.weight"].clone();
            let ids: Vec<u32> = (0..cfg.vocab_size as u32).step_by(2).collect();
            let idx = Tensor::new(ids.as_slice(), &Device::Cpu).unwrap();
            ts.push((
                format!("{gp}.nextn.draft_head.weight"),
                lm.index_select(&idx, 0).unwrap(),
            ));
            let ids: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
            ts.push((
                format!("{gp}.nextn.draft_head_ids.weight"),
                Tensor::new(ids, &Device::Cpu).unwrap(),
            ));
        }
    }
    ts.push(("token_embd.weight".into(), hf.get("embed_tokens.weight")));
    ts.push(("output_norm.weight".into(), plus1(hf.get("norm.weight"))));
    let tied = cfg.tie_word_embeddings;
    if !tied {
        ts.push(("output.weight".into(), hf.t["lm_head.weight"].clone()));
    }

    let mut md: Vec<(String, Value)> = vec![];
    let mut kv = |k: &str, v: Value| md.push((format!("qwen35.{k}"), v));
    kv("block_count", u32v(n + n_mtp));
    if n_mtp > 0 {
        kv("nextn_predict_layers", u32v(n_mtp));
    }
    kv("embedding_length", u32v(cfg.hidden_size));
    kv("feed_forward_length", u32v(cfg.intermediate_size));
    kv("attention.head_count", u32v(cfg.num_attention_heads));
    kv("attention.head_count_kv", u32v(cfg.num_key_value_heads));
    kv("attention.key_length", u32v(cfg.head_dim));
    kv("attention.value_length", u32v(cfg.head_dim));
    kv(
        "attention.layer_norm_rms_epsilon",
        Value::F32(cfg.rms_norm_eps as f32),
    );
    let theta = cfg.rope_parameters.as_ref().and_then(|r| r.rope_theta);
    kv(
        "rope.freq_base",
        Value::F32(theta.or(cfg.rope_theta).unwrap() as f32),
    );
    let factor = cfg
        .rope_parameters
        .as_ref()
        .and_then(|r| r.partial_rotary_factor)
        .or(cfg.partial_rotary_factor)
        .unwrap();
    kv(
        "rope.dimension_count",
        u32v((cfg.head_dim as f64 * factor) as usize),
    );
    kv("context_length", u32v(cfg.max_position_embeddings));
    if let Some(section) = cfg
        .rope_parameters
        .as_ref()
        .and_then(|r| r.mrope_section.clone())
    {
        let mut v: Vec<Value> = section.iter().map(|&s| Value::I32(s as i32)).collect();
        v.push(Value::I32(0));
        kv("rope.dimension_sections", Value::Array(v));
    }
    kv("ssm.conv_kernel", u32v(cfg.linear_conv_kernel_dim));
    kv("ssm.inner_size", u32v(nv * dv));
    kv("ssm.state_size", u32v(dk));
    kv("ssm.time_step_rank", u32v(nv));
    kv("ssm.group_count", u32v(nk));
    let recurrent: Vec<Value> = types
        .iter()
        .map(|t| Value::Bool(*t == qwen3_5::LayerType::LinearAttention))
        .chain((0..n_mtp).map(|_| Value::Bool(false)))
        .collect();
    kv("attention.recurrent_layers", Value::Array(recurrent));
    md.push((
        "general.architecture".into(),
        Value::String("qwen35".into()),
    ));

    if let Some(f) = opts.fold {
        // Fold every 2D matmul weight but the token table, then rotate the token table.
        let is_folded = |n: &str| {
            let kinds = [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_qkv",
                "attn_gate",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "ssm_out",
            ];
            n == "output.weight" || kinds.iter().any(|k| n.ends_with(&format!(".{k}.weight")))
        };
        let mut names = vec![];
        let mut widths = std::collections::BTreeSet::new();
        for (n, t) in ts.iter_mut() {
            if is_folded(n) {
                widths.insert(t.dim(1).unwrap());
                *t = fold_weight(t, f);
                names.push(Value::String(n.clone()));
            }
        }
        for (n, t) in ts.iter_mut() {
            if n == "token_embd.weight" {
                widths.insert(t.dim(1).unwrap());
                *t = fold_weight(t, f);
            }
        }
        let p = |k: &str| format!("prism.hadamard.{k}");
        md.push((p("version"), u32v(if f.tied { 2 } else { 1 })));
        md.push((p("tied_output"), Value::Bool(f.tied)));
        md.push((p("block_size"), u32v(f.block)));
        let transform = "normalized-sylvester-walsh-hadamard";
        md.push((p("transform"), Value::String(transform.into())));
        md.push((p("axis"), Value::String("input-last-dimension".into())));
        let mode = if f.signs { "explicit" } else { "identity" };
        md.push((p("sign_mode"), Value::String(mode.into())));
        if f.signs {
            let ws: Vec<Value> = widths.iter().map(|&w| Value::I32(w as i32)).collect();
            let vs: Vec<Value> = widths
                .iter()
                .flat_map(|&w| signs(w))
                .map(Value::I32)
                .collect();
            md.push((p("sign_widths"), Value::Array(ws)));
            md.push((p("sign_values"), Value::Array(vs)));
        }
        md.push((p("weight_names"), Value::Array(names)));
        let inverse = vec![Value::String("token_embd.weight".into())];
        md.push((p("inverse_weight_names"), Value::Array(inverse)));
        md.push((p("gdn_v_grouped"), Value::Bool(true)));
    }
    if opts.scales {
        let mut extra = vec![];
        for (n, t) in ts.iter_mut() {
            if n.ends_with(".ffn_down.weight") || n.ends_with(".attn_output.weight") {
                *t = (t.clone() / 2.).unwrap();
                let base = n.trim_end_matches(".weight");
                extra.push((
                    format!("{base}.scale"),
                    Tensor::new(&[2f32], &Device::Cpu).unwrap(),
                ));
            }
        }
        ts.extend(extra);
    }

    let qs: Vec<(String, QTensor)> = ts
        .into_iter()
        .map(|(n, t)| {
            let dtype = match opts.dtype {
                Some(d) if t.rank() == 2 && !n.contains("conv1d") => d,
                _ => GgmlDType::F32,
            };
            (n, QTensor::quantize(&t, dtype).unwrap())
        })
        .collect();
    let mdr: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tsr: Vec<(&str, &QTensor)> = qs.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut buf = Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &mdr, &tsr).unwrap();
    buf.into_inner()
}

/// The model with transformers' query-key norm, to compare with its references.
pub fn model(bytes: Vec<u8>) -> Model {
    model_with(bytes, L2Norm::Sqrt)
}

pub fn model_with(bytes: Vec<u8>, l2_norm: L2Norm) -> Model {
    let mut r = Cursor::new(bytes);
    let ct = gguf_file::Content::read(&mut r).unwrap();
    let mut cfg = Config::from_gguf(&ct).unwrap();
    assert_eq!(cfg.l2_norm, L2Norm::Max);
    cfg.l2_norm = l2_norm;
    Model::new(&cfg, &ct, &mut r, &Device::Cpu).unwrap()
}

/// Writes the `mmproj` GGUF file of the vision tower of a multimodal checkpoint, as llama.cpp's
/// converter does (`conversion/qwen3vl.py`): the patch convolution split into one 2D convolution
/// per frame, the blocks under `v.blk`, and the merger under `v.post_ln` and `mm`.
pub fn mmproj(variant: &str) -> Vec<u8> {
    let t = load(&format!("qwen3_5/{variant}/model.safetensors")).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&read(&format!("qwen3_5/{variant}/config.json"))).unwrap();
    let v = &json["vision_config"];
    let n = |k: &str| v[k].as_u64().unwrap() as usize;
    let g = |k: &str| t[&format!("model.visual.{k}")].clone();
    let mut ts: Vec<(String, Tensor)> = vec![];
    let w = g("patch_embed.proj.weight");
    ts.push((
        "v.patch_embd.weight".into(),
        w.narrow(2, 0, 1).unwrap().squeeze(2).unwrap(),
    ));
    ts.push((
        "v.patch_embd.weight.1".into(),
        w.narrow(2, 1, 1).unwrap().squeeze(2).unwrap(),
    ));
    ts.push(("v.patch_embd.bias".into(), g("patch_embed.proj.bias")));
    ts.push(("v.position_embd.weight".into(), g("pos_embed.weight")));
    for i in 0..n("depth") {
        for (gn, h) in [
            ("ln1", "norm1"),
            ("ln2", "norm2"),
            ("attn_qkv", "attn.qkv"),
            ("attn_out", "attn.proj"),
            ("ffn_up", "mlp.linear_fc1"),
            ("ffn_down", "mlp.linear_fc2"),
        ] {
            for p in ["weight", "bias"] {
                ts.push((
                    format!("v.blk.{i}.{gn}.{p}"),
                    g(&format!("blocks.{i}.{h}.{p}")),
                ));
            }
        }
    }
    for (gn, h) in [
        ("v.post_ln", "merger.norm"),
        ("mm.0", "merger.linear_fc1"),
        ("mm.2", "merger.linear_fc2"),
    ] {
        for p in ["weight", "bias"] {
            ts.push((format!("{gn}.{p}"), g(&format!("{h}.{p}"))));
        }
    }
    let side = (n("num_position_embeddings") as f64).sqrt() as usize;
    let f3 = |x: f32| Value::Array(vec![Value::F32(x); 3]);
    let md: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String("clip".into())),
        (
            "clip.projector_type".into(),
            Value::String("qwen3vl_merger".into()),
        ),
        ("clip.has_vision_encoder".into(), Value::Bool(true)),
        ("clip.use_gelu".into(), Value::Bool(true)),
        ("clip.vision.block_count".into(), u32v(n("depth"))),
        (
            "clip.vision.embedding_length".into(),
            u32v(n("hidden_size")),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            u32v(n("intermediate_size")),
        ),
        (
            "clip.vision.attention.head_count".into(),
            u32v(n("num_heads")),
        ),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            Value::F32(1e-6),
        ),
        ("clip.vision.patch_size".into(), u32v(n("patch_size"))),
        (
            "clip.vision.image_size".into(),
            u32v(side * n("patch_size")),
        ),
        (
            "clip.vision.spatial_merge_size".into(),
            u32v(n("spatial_merge_size")),
        ),
        (
            "clip.vision.projection_dim".into(),
            u32v(n("out_hidden_size")),
        ),
        ("clip.vision.image_mean".into(), f3(0.5)),
        ("clip.vision.image_std".into(), f3(0.5)),
    ];
    let qs: Vec<(String, QTensor)> = ts
        .into_iter()
        .map(|(n, t)| (n, QTensor::quantize(&t, GgmlDType::F32).unwrap()))
        .collect();
    let mdr: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tsr: Vec<(&str, &QTensor)> = qs.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut buf = Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &mdr, &tsr).unwrap();
    buf.into_inner()
}
