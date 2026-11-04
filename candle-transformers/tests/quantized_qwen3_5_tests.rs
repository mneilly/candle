//! The `qwen35` GGUF model against the Qwen3.5 references. The GGUF files are written here from
//! the safetensors fixtures as llama.cpp's converter does (PrismML-Eng/llama.cpp,
//! `conversion/qwen.py`): renamed tensors, norm weights holding `1 + w`, `ssm_a = -exp(A_log)`,
//! the Gated DeltaNet value heads reordered from grouped to tiled, and the MTP head as `nextn`
//! layers after the trunk. Variants fold Hadamard rotations into the weights, quantize to the
//! ternary types, fuse the attention projections, add per-tensor scales and a draft head.

mod common;

use candle::quantized::gguf_file::{self, Value};
use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3_5::{Config, L2Norm, Model};
use candle_transformers::models::qwen3_5::{self, MtpModel};
use common::{assert_close, load, read};
use std::collections::HashMap;
use std::io::Cursor;

/// A Hadamard fold: every 2D weight except the GDN projections to the value heads, the token
/// table rotated for lookups, and with `tied` the output head bound to it (version 2).
#[derive(Clone, Copy)]
struct Fold {
    block: usize,
    signs: bool,
    tied: bool,
}

#[derive(Clone, Copy, Default)]
struct Opts {
    fold: Option<Fold>,
    fused_qkv: bool,
    /// Halve `ffn_down` and `attn_output`, with a scale of 2.
    scales: bool,
    /// A draft head over every other token.
    draft_head: bool,
    /// MTP layers with their own copies of the token table and LM head.
    mtp_own_head: bool,
    dtype: Option<GgmlDType>,
}

struct Hf {
    t: HashMap<String, Tensor>,
    cfg: qwen3_5::Config,
    /// The prefix of the trunk weights, `model.` or `model.language_model.`.
    prefix: String,
}

impl Hf {
    fn fixture(variant: &str) -> Self {
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

    fn get(&self, name: &str) -> Tensor {
        match self.t.get(&format!("{}{name}", self.prefix)) {
            Some(t) => t.clone(),
            None => self.t[name].clone(),
        }
    }
}

/// Grouped value heads `g * rep + r` to tiled `r * n_k + g` along `dim`.
fn tile(t: &Tensor, dim: usize, nk: usize, rep: usize, hd: usize) -> Tensor {
    let mut shape = t.dims().to_vec();
    let mut split = shape.clone();
    split.splice(dim..dim + 1, [nk, rep, hd]);
    let t = t.reshape(split).unwrap().transpose(dim, dim + 1).unwrap();
    shape[dim] = nk * rep * hd;
    t.contiguous().unwrap().reshape(shape).unwrap()
}

/// `H[i, j] = (-1)^popcount(i & j) / sqrt(n)`.
fn hadamard(n: usize) -> Tensor {
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
fn signs(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| if (i * 7 + i / 3) % 5 < 2 { -1 } else { 1 })
        .collect()
}

/// `W D H`: the signs then the block rotation on the input dimension.
fn fold_weight(w: &Tensor, f: Fold) -> Tensor {
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

fn u32v(v: usize) -> Value {
    Value::U32(v as u32)
}

/// Writes the GGUF file of a Qwen3.5 checkpoint.
fn convert(hf: &Hf, opts: Opts) -> Vec<u8> {
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
fn model(bytes: Vec<u8>) -> Model {
    model_with(bytes, L2Norm::Sqrt)
}

fn model_with(bytes: Vec<u8>, l2_norm: L2Norm) -> Model {
    let mut r = Cursor::new(bytes);
    let ct = gguf_file::Content::read(&mut r).unwrap();
    let mut cfg = Config::from_gguf(&ct).unwrap();
    assert_eq!(cfg.l2_norm, L2Norm::Max);
    cfg.l2_norm = l2_norm;
    Model::new(&cfg, &ct, &mut r, &Device::Cpu).unwrap()
}

/// Prefill, cached decode steps and a continuation against the transformers references.
fn check_logits(model: &mut Model, variant: &str, tol: f32) {
    let r = load(&format!("qwen3_5/{variant}/reference.safetensors")).unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let p = prompt.dim(1).unwrap();
    model.clear_kv_cache();
    assert_close(
        &model.forward_all(&prompt, 0).unwrap(),
        &r["prefill_logits"],
        tol,
        "prefill",
    );
    let toks = r["decode_tokens"].to_dtype(DType::U32).unwrap();
    for i in 0..toks.dim(1).unwrap() {
        let l = model
            .forward_all(&toks.narrow(1, i, 1).unwrap(), p + i)
            .unwrap();
        assert_close(&l, &r[&format!("decode_logits.{i}")], tol, "decode");
    }
    let cont = r["cont"].to_dtype(DType::U32).unwrap();
    let l = model.forward_all(&cont, p + toks.dim(1).unwrap()).unwrap();
    assert_close(&l, &r["cont_logits"], tol, "continuation");
    model.clear_kv_cache();
    let last = model.forward(&prompt, 0).unwrap();
    let expected = r["prefill_logits"].narrow(1, p - 1, 1).unwrap();
    assert_close(&last, &expected, tol, "forward");
}

fn check_mtp(model: &mut Model, tol: f32) {
    let r = load("qwen3_5/text/mtp_reference.safetensors").unwrap();
    let prompt = load("qwen3_5/text/reference.safetensors").unwrap()["prompt"]
        .to_dtype(DType::U32)
        .unwrap();
    let p = prompt.dim(1).unwrap();
    model.clear_kv_cache();
    let hidden = model.forward_hidden(&prompt, 0).unwrap();
    assert_close(&hidden, &r["main_hidden"], tol, "main hidden");
    let g = model
        .mtp_forward_hidden(
            &prompt.narrow(1, 1, p - 1).unwrap(),
            &hidden.narrow(1, 0, p - 1).unwrap(),
            0,
            0,
        )
        .unwrap();
    assert_close(&g, &r["teacher_hidden"], tol, "teacher hidden");
    assert_close(
        &model.mtp_logits(&g, 0).unwrap(),
        &r["teacher_logits"],
        tol,
        "teacher logits",
    );
    let inputs = r["draft_inputs"].to_dtype(DType::U32).unwrap();
    let mut h = hidden.narrow(1, p - 1, 1).unwrap();
    for i in 0..inputs.dim(1).unwrap() {
        h = model
            .mtp_forward_hidden(&inputs.narrow(1, i, 1).unwrap(), &h, p - 1 + i, i)
            .unwrap();
        let l = model.mtp_logits(&h, i).unwrap();
        assert_close(&l, &r[&format!("draft_logits.{i}")], tol, "draft");
    }
}

#[test]
fn matches_the_reference() {
    let hf = Hf::fixture("text");
    let bytes = convert(&hf, Opts::default());
    let mut r = Cursor::new(bytes.clone());
    let ct = gguf_file::Content::read(&mut r).unwrap();
    let cfg = Config::from_gguf(&ct).unwrap();
    assert_eq!((cfg.num_hidden_layers, cfg.num_mtp_layers), (4, 1));
    assert_eq!(cfg.recurrent, [true, true, true, false]);
    assert_eq!((cfg.head_k_dim, cfg.head_v_dim, cfg.rope_dim), (8, 8, 8));
    let mut m = model(bytes);
    assert!(m.has_mtp() && !m.has_draft_head());
    check_logits(&mut m, "text", 1e-4);
    check_mtp(&mut m, 1e-4);
}

#[test]
fn multimodal_layout_without_mtp() {
    // Tied embeddings and no MTP layers.
    let mut m = model(convert(&Hf::fixture("multimodal"), Opts::default()));
    assert!(!m.has_mtp());
    check_logits(&mut m, "multimodal", 1e-4);
    let ids = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
    let h = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
    let err = m
        .mtp_forward_hidden(&ids, &h, 0, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no MTP layers"), "{err}");
}

#[test]
fn fused_attention_and_scales() {
    let opts = Opts {
        fused_qkv: true,
        scales: true,
        ..Opts::default()
    };
    let mut m = model(convert(&Hf::fixture("text"), opts));
    check_logits(&mut m, "text", 1e-4);
    check_mtp(&mut m, 1e-4);
}

#[test]
fn hadamard_folded_weights() {
    for (block, signs) in [(8, false), (16, true)] {
        let fold = Fold {
            block,
            signs,
            tied: false,
        };
        let opts = Opts {
            fold: Some(fold),
            ..Opts::default()
        };
        let mut m = model(convert(&Hf::fixture("text"), opts));
        check_logits(&mut m, "text", 1e-4);
        check_mtp(&mut m, 1e-4);
    }
    // Version 2: the output head is the rotated token table.
    let fold = Fold {
        block: 16,
        signs: true,
        tied: true,
    };
    let opts = Opts {
        fold: Some(fold),
        ..Opts::default()
    };
    let mut m = model(convert(&Hf::fixture("multimodal"), opts));
    check_logits(&mut m, "multimodal", 1e-4);
}

fn greedy(model: &mut Model, prompt: &Tensor, n: usize) -> Vec<u32> {
    model.clear_kv_cache();
    let p = prompt.dim(1).unwrap();
    let mut logits = model.forward(prompt, 0).unwrap();
    let mut tokens = vec![];
    for i in 0..n {
        let t = logits.flatten_all().unwrap().argmax(0).unwrap();
        tokens.push(t.to_scalar::<u32>().unwrap());
        logits = model.forward(&t.reshape((1, 1)).unwrap(), p + i).unwrap();
    }
    tokens
}

#[test]
fn speculative_decoding_and_draft_head() {
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let prompt = prompt.narrow(0, 0, 1).unwrap();
    for draft_head in [false, true] {
        let opts = Opts {
            draft_head,
            ..Opts::default()
        };
        let mut m = model(convert(&Hf::fixture("text"), opts));
        assert_eq!(m.has_draft_head(), draft_head);
        let expected = greedy(&mut m, &prompt, 16);
        for drafts in 1..=2 {
            let mut tokens = vec![];
            let stats = m
                .generate_mtp(&prompt, 16, drafts, |t| {
                    tokens.push(t);
                    Ok(true)
                })
                .unwrap();
            assert_eq!(tokens, expected, "draft head {draft_head}, {drafts} drafts");
            assert_eq!(stats.drafted, stats.rounds * drafts);
        }
        // The draft head proposes the best of its own tokens, the even ones.
        m.clear_kv_cache();
        let h = m.forward_hidden(&prompt, 0).unwrap();
        let p = prompt.dim(1).unwrap();
        let g = m
            .mtp_forward_hidden(
                &prompt.narrow(1, 1, p - 1).unwrap(),
                &h.narrow(1, 0, p - 1).unwrap(),
                0,
                0,
            )
            .unwrap()
            .narrow(1, p - 2, 1)
            .unwrap();
        let logits = m.mtp_logits(&g, 0).unwrap().flatten_all().unwrap();
        let logits = logits.to_vec1::<f32>().unwrap();
        let best = |step: usize| {
            (0..logits.len())
                .step_by(step)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap() as u32
        };
        let expected = if draft_head { best(2) } else { best(1) };
        assert_eq!(MtpModel::draft_token(&m, &g).unwrap(), expected);
    }
}

#[test]
fn cache_snapshots() {
    let mut m = model(convert(&Hf::fixture("text"), Opts::default()));
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let cont = r["cont"].to_dtype(DType::U32).unwrap();
    let p = prompt.dim(1).unwrap();
    m.forward(&prompt, 0).unwrap();
    let snapshot = m.cache();
    let first = m.forward_all(&cont, p).unwrap();
    m.forward_all(&cont, p + cont.dim(1).unwrap()).unwrap();
    m.set_cache(snapshot).unwrap();
    assert_close(&m.forward_all(&cont, p).unwrap(), &first, 0., "rolled back");
    let other = model(convert(&Hf::fixture("multimodal"), Opts::default()));
    let err = m.set_cache(other.cache()).unwrap_err().to_string();
    assert!(err.contains("another model"), "{err}");
}

/// A random Qwen3.5 checkpoint whose matmul inputs are multiples of 128, with ternary weights
/// `{-1, 0, 1} * scale` per 128-wide block, as the ternary types store them exactly.
fn ternary_checkpoint() -> Hf {
    let json = r#"{"vocab_size": 64, "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 64, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "full_attention_interval": 2,
        "mtp_num_hidden_layers": 1, "rope_theta": 10000.0, "partial_rotary_factor": 0.25}"#;
    let cfg = qwen3_5::Config::from_json(json).unwrap();
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    // A var map only holds the variables asked for: ask for the MTP head first so that the
    // model loads it.
    vb.get((128, 256), "mtp.fc.weight").unwrap();
    qwen3_5::Model::new(&cfg, vb).unwrap();
    let mut t = HashMap::new();
    let mut seed = 1u64;
    for (name, var) in varmap.data().lock().unwrap().iter() {
        let v = var.as_tensor();
        let n = v.elem_count();
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let x: Vec<f32> = if v.rank() == 2 && v.dim(1).unwrap() % 128 == 0 {
            // Ternary with a power-of-two scale and a +-scale in every block.
            (0..n)
                .map(|i| {
                    let t = if i % 128 == 0 {
                        1.
                    } else {
                        (next() * 3.).floor() - 1.
                    };
                    t * 0.0625
                })
                .collect()
        } else if name.contains("norm") {
            (0..n).map(|_| (next() - 0.5) * 0.2).collect()
        } else {
            (0..n).map(|_| (next() - 0.5) * 0.5).collect()
        };
        t.insert(
            name.clone(),
            Tensor::from_vec(x, v.dims(), &Device::Cpu).unwrap(),
        );
    }
    // The MTP head is loaded when its weights exist.
    assert!(t.contains_key("mtp.fc.weight"));
    Hf {
        t,
        cfg,
        prefix: "model.".to_string(),
    }
}

#[test]
fn ternary_types_match_the_unquantized_model() {
    let hf = ternary_checkpoint();
    let vb = candle_nn::VarBuilder::from_tensors(hf.t.clone(), DType::F32, &Device::Cpu);
    let mut reference = qwen3_5::Model::new(&hf.cfg, vb).unwrap();
    let ids: Vec<u32> = (0..20).map(|i| (i * 7 % 64) as u32).collect();
    let ids = Tensor::new(ids.as_slice(), &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let expected = reference.forward_all(&ids, 0).unwrap();
    reference.clear_kv_cache();
    let h = reference.forward_hidden(&ids, 0).unwrap();
    let pairs = (ids.narrow(1, 1, 19).unwrap(), h.narrow(1, 0, 19).unwrap());
    let expected_mtp = reference
        .mtp_forward_hidden(&pairs.0, &pairs.1, 0, 0)
        .unwrap();
    for dtype in [GgmlDType::Q2_0, GgmlDType::PQ2_0, GgmlDType::PTQ1_0] {
        let opts = Opts {
            dtype: Some(dtype),
            ..Opts::default()
        };
        let mut m = model(convert(&hf, opts));
        // Ternary weights are stored exactly: only the activations are rounded, to Q8_0, which
        // on this small random model moves the logits by about 2% (as for Q8_0 weights).
        let what = format!("{dtype:?}");
        assert_close(&m.forward_all(&ids, 0).unwrap(), &expected, 0.05, &what);
        m.clear_kv_cache();
        let h = m.forward_hidden(&ids, 0).unwrap();
        let pairs = (ids.narrow(1, 1, 19).unwrap(), h.narrow(1, 0, 19).unwrap());
        let g = m.mtp_forward_hidden(&pairs.0, &pairs.1, 0, 0).unwrap();
        assert_close(&g, &expected_mtp, 0.05, &what);
    }
    // A fold over 128-wide blocks with signs, unquantized: a rotated ternary weight is no
    // longer ternary.
    let opts = Opts {
        fold: Some(Fold {
            block: 128,
            signs: true,
            tied: false,
        }),
        ..Opts::default()
    };
    let mut m = model(convert(&hf, opts));
    assert_close(&m.forward_all(&ids, 0).unwrap(), &expected, 1e-4, "folded");
}

#[test]
fn ggml_l2_norm_is_close_to_transformers() {
    // PrismML's packed models expect ggml's `x / max(|x|, eps)`, which differs from
    // transformers' `x / sqrt(|x|^2 + eps)` for small queries and keys only.
    let bytes = convert(&Hf::fixture("text"), Opts::default());
    let mut max = model_with(bytes.clone(), L2Norm::Max);
    let mut sqrt = model_with(bytes, L2Norm::Sqrt);
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let a = max.forward_all(&prompt, 0).unwrap();
    let b = sqrt.forward_all(&prompt, 0).unwrap();
    let d = common::rel_diff(&a, &b).unwrap();
    assert!(d > 0. && d < 1e-2, "{d}");
}

#[test]
fn invalid_files() {
    let hf = Hf::fixture("text");
    let bytes = convert(&hf, Opts::default());
    let edit = |f: &dyn Fn(&mut HashMap<String, Value>)| {
        let mut r = Cursor::new(bytes.clone());
        let mut ct = gguf_file::Content::read(&mut r).unwrap();
        f(&mut ct.metadata);
        Model::from_gguf(&ct, &mut r, &Device::Cpu)
            .unwrap_err()
            .to_string()
    };
    let err = edit(&|md| {
        md.insert("general.architecture".into(), Value::String("llama".into()));
    });
    assert!(err.contains("expected a qwen35"), "{err}");
    let err = edit(&|md| {
        md.remove("qwen35.embedding_length");
    });
    assert!(err.contains("qwen35.embedding_length"), "{err}");
    let err = edit(&|md| {
        md.insert("prism.hadamard.version".into(), Value::U32(3));
    });
    assert!(err.contains("unsupported prism.hadamard.version"), "{err}");
    let err = edit(&|md| {
        md.insert(
            "qwen35.attention.recurrent_layers".into(),
            Value::Array(vec![Value::Bool(true)]),
        );
    });
    assert!(err.contains("recurrent flags"), "{err}");
    let err = edit(&|md| {
        md.insert("qwen35.embedding_length".into(), Value::String("32".into()));
    });
    assert!(err.contains("not an integer"), "{err}");
    let mut r = Cursor::new(bytes.clone());
    let mut ct = gguf_file::Content::read(&mut r).unwrap();
    ct.tensor_infos.remove("token_embd.weight");
    let err = Config::from_gguf(&ct).unwrap_err().to_string();
    assert!(err.contains("token_embd.weight"), "{err}");
}

/// Rewrites the F32 tensor `name` of a GGUF file as I32, in place: both take 4 bytes.
fn to_i32(mut bytes: Vec<u8>, name: &str) -> Vec<u8> {
    let mut r = Cursor::new(bytes.clone());
    let ct = gguf_file::Content::read(&mut r).unwrap();
    let info = &ct.tensor_infos[name];
    let start = (ct.tensor_data_offset + info.offset) as usize;
    for i in 0..info.shape.elem_count() {
        let at = start + 4 * i;
        let v = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as i32;
        bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    // The tensor info: the name, the dimension count and dimensions, then the type.
    let key = [&(name.len() as u64).to_le_bytes()[..], name.as_bytes()].concat();
    let pos = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len();
    let n_dims = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    let ty = pos + 4 + 8 * n_dims;
    bytes[ty..ty + 4].copy_from_slice(&26u32.to_le_bytes());
    bytes
}

#[test]
fn mtp_layer_variants() {
    // Own copies of the token table and LM head give the same outputs.
    let opts = Opts {
        mtp_own_head: true,
        ..Opts::default()
    };
    let mut m = model(convert(&Hf::fixture("text"), opts));
    check_mtp(&mut m, 1e-4);

    // Draft ids stored as I32, the type llama.cpp indexes with.
    let opts = Opts {
        draft_head: true,
        ..Opts::default()
    };
    let bytes = convert(&Hf::fixture("text"), opts);
    let a = model(bytes.clone());
    let b = model(to_i32(bytes, "blk.4.nextn.draft_head_ids.weight"));
    assert!(b.has_draft_head());
    let h: Vec<f32> = (0..32).map(|i| (i as f32 * 0.7).sin()).collect();
    let h = Tensor::from_vec(h, (1, 1, 32), &Device::Cpu).unwrap();
    assert_eq!(
        MtpModel::draft_token(&a, &h).unwrap(),
        MtpModel::draft_token(&b, &h).unwrap()
    );

    // Without its own head norm an MTP layer uses the final norm: with the two norms equal, the
    // outputs are the same.
    let mut hf = Hf::fixture("text");
    let norm = hf.get("norm.weight");
    hf.t.insert("mtp.norm.weight".into(), norm);
    let bytes = convert(&hf, Opts::default());
    let mut with_norm = model(bytes.clone());
    let mut r = Cursor::new(bytes);
    let mut ct = gguf_file::Content::read(&mut r).unwrap();
    ct.tensor_infos
        .remove("blk.4.nextn.shared_head_norm.weight");
    let mut cfg = Config::from_gguf(&ct).unwrap();
    cfg.l2_norm = L2Norm::Sqrt;
    let mut without = Model::new(&cfg, &ct, &mut r, &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[3u32, 1, 4, 1, 5]], &Device::Cpu).unwrap();
    let hs = with_norm.forward_hidden(&ids, 0).unwrap();
    let a = with_norm.mtp_forward_hidden(&ids, &hs, 0, 0).unwrap();
    let b = without.mtp_forward_hidden(&ids, &hs, 0, 0).unwrap();
    assert_close(&a, &b, 0., "final norm");
}

#[test]
fn layer_types_from_the_interval() {
    // Without the recurrent-layer flags, every fourth layer is full attention.
    let bytes = convert(&Hf::fixture("text"), Opts::default());
    let mut r = Cursor::new(bytes);
    let mut ct = gguf_file::Content::read(&mut r).unwrap();
    ct.metadata.remove("qwen35.attention.recurrent_layers");
    let mut cfg = Config::from_gguf(&ct).unwrap();
    assert_eq!(cfg.recurrent, [true, true, true, false]);
    ct.metadata
        .insert("qwen35.full_attention_interval".into(), Value::U32(2));
    assert_eq!(
        Config::from_gguf(&ct).unwrap().recurrent,
        [true, false, true, false]
    );
    cfg.l2_norm = L2Norm::Sqrt;
    let mut m = Model::new(&cfg, &ct, &mut r, &Device::Cpu).unwrap();
    check_logits(&mut m, "text", 1e-4);
}

#[test]
fn invalid_hadamard_metadata() {
    let fold = Fold {
        block: 16,
        signs: true,
        tied: false,
    };
    let opts = Opts {
        fold: Some(fold),
        ..Opts::default()
    };
    let bytes = convert(&Hf::fixture("text"), opts);
    let p = |k: &str| format!("prism.hadamard.{k}");
    let s = |v: &str| Value::String(v.into());
    let ints = |v: &[i32]| Value::Array(v.iter().map(|&i| Value::I32(i)).collect());
    // Each case sets (or with `None` removes) metadata keys, and names the expected error.
    type Edits = Vec<(String, Option<Value>)>;
    let cases: Vec<(Edits, &str)> = vec![
        (
            vec![(p("tied_output"), Some(Value::Bool(true)))],
            "version 2 requires tied_output",
        ),
        (
            vec![(p("block_size"), None)],
            "cannot find prism.hadamard.block_size",
        ),
        (
            vec![(p("block_size"), Some(Value::U32(12)))],
            "invalid prism.hadamard.block_size",
        ),
        (
            vec![(p("transform"), Some(s("dct")))],
            "unsupported prism.hadamard.transform",
        ),
        (
            vec![(p("transform"), None)],
            "cannot find prism.hadamard.transform",
        ),
        (
            vec![(p("axis"), Some(s("output")))],
            "unsupported prism.hadamard.axis",
        ),
        (
            vec![(p("sign_mode"), Some(s("random")))],
            "unsupported prism.hadamard.sign_mode",
        ),
        (vec![(p("sign_widths"), None)], "without sign widths"),
        (
            vec![(p("sign_widths"), Some(ints(&[12])))],
            "invalid prism.hadamard sign width",
        ),
        (
            vec![(p("sign_widths"), Some(ints(&[16, 16])))],
            "duplicate prism.hadamard sign width",
        ),
        (
            vec![(p("sign_values"), Some(ints(&[2; 16])))],
            "invalid prism.hadamard sign width",
        ),
        (
            vec![
                (p("sign_widths"), Some(ints(&[16]))),
                (p("sign_values"), Some(ints(&[2; 16]))),
            ],
            "must be +-1",
        ),
        (
            vec![
                (p("sign_widths"), Some(ints(&[16]))),
                (p("sign_values"), Some(ints(&[1; 17]))),
            ],
            "length mismatch",
        ),
        (
            vec![
                (p("sign_widths"), Some(ints(&[16]))),
                (p("sign_values"), Some(ints(&[1; 16]))),
            ],
            "no sign vector of width",
        ),
        (
            vec![(p("weight_names"), Some(Value::Array(vec![])))],
            "weight_names is empty",
        ),
        (
            vec![(
                p("inverse_weight_names"),
                Some(Value::Array(vec![s("output.weight")])),
            )],
            "only inverts the token embedding",
        ),
        (
            vec![
                (p("version"), Some(Value::U32(2))),
                (p("tied_output"), Some(Value::Bool(true))),
                (p("inverse_weight_names"), Some(Value::Array(vec![]))),
            ],
            "requires a rotated token embedding",
        ),
        (
            vec![
                (p("version"), Some(Value::U32(2))),
                (p("tied_output"), Some(Value::Bool(true))),
            ],
            "requires output.weight to be absent",
        ),
        (
            vec![
                (p("block_size"), Some(Value::U32(64))),
                (p("sign_mode"), Some(s("identity"))),
            ],
            "does not divide the width",
        ),
    ];
    for (edits, expected) in cases {
        let mut r = Cursor::new(bytes.clone());
        let mut ct = gguf_file::Content::read(&mut r).unwrap();
        for (k, v) in edits {
            match v {
                Some(v) => ct.metadata.insert(k, v),
                None => ct.metadata.remove(&k),
            };
        }
        let err = Model::from_gguf(&ct, &mut r, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{expected}: {err}");
    }
}
