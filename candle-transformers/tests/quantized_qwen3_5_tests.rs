//! The `qwen35` GGUF model against the Qwen3.5 references. The GGUF files are written here from
//! the safetensors fixtures as llama.cpp's converter does (PrismML-Eng/llama.cpp,
//! `conversion/qwen.py`): renamed tensors, norm weights holding `1 + w`, `ssm_a = -exp(A_log)`,
//! the Gated DeltaNet value heads reordered from grouped to tiled, and the MTP head as `nextn`
//! layers after the trunk. Variants fold Hadamard rotations into the weights, quantize to the
//! ternary types, fuse the attention projections, add per-tensor scales and a draft head.

mod common;

use candle::quantized::gguf_file::{self, Value};
use candle::quantized::GgmlDType;
use candle::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3_5::{Config, L2Norm, Model};
use candle_transformers::models::qwen3_5::{self, MtpModel};
use common::qwen35_gguf::*;
use common::{assert_close, load};
use std::collections::HashMap;
use std::io::Cursor;

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

#[test]
fn images_through_the_mmproj_file() {
    use candle_transformers::models::qwen3_5_vl::{Images, VisionModel};
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let mut m = model(convert(&Hf::fixture("vision"), Opts::default()));
    let mut f = Cursor::new(mmproj("vision"));
    let ct = gguf_file::Content::read(&mut f).unwrap();
    let (vision, processor) = VisionModel::from_mmproj(&ct, &mut f, &Device::Cpu).unwrap();
    assert_eq!(processor.factor(), 8);
    let (mut pixels, mut grids) = (vec![], vec![]);
    for i in 0..2 {
        let (p, g) = processor.preprocess(&r[&format!("image.{i}")]).unwrap();
        pixels.push(p);
        grids.push(g);
    }
    let pixels = Tensor::cat(&pixels, 0).unwrap();
    assert_close(
        &vision.forward(&pixels, &grids).unwrap(),
        &r["vision_out"],
        1e-4,
        "vision",
    );
    let ids = r["input_ids"].to_dtype(DType::U32).unwrap();
    let images = Images {
        pixel_values: &pixels,
        grids: &grids,
    };
    let h = m
        .forward_prompt_hidden(&ids, Some((&vision, images)), 45)
        .unwrap();
    assert_close(
        &m.logits(&h).unwrap(),
        &r["prefill_logits"],
        1e-4,
        "prefill",
    );
    assert_eq!(m.rope_delta(), -3);
    let l = ids.dim(1).unwrap();
    let toks = r["decode_tokens"].to_dtype(DType::U32).unwrap();
    for i in 0..toks.dim(1).unwrap() {
        let logits = m.forward(&toks.narrow(1, i, 1).unwrap(), l + i).unwrap();
        assert_close(&logits, &r[&format!("decode_logits.{i}")], 1e-4, "decode");
    }
    // Without images the prompt is text.
    let a = m.forward_prompt_hidden(&ids, None, 45).unwrap();
    assert_eq!(m.rope_delta(), 0);
    m.clear_kv_cache();
    assert_close(&a, &m.forward_hidden(&ids, 0).unwrap(), 0., "text");

    // Other projectors and deepstack layers are refused.
    let mut f = Cursor::new(mmproj("vision"));
    let mut ct = gguf_file::Content::read(&mut f).unwrap();
    ct.metadata
        .insert("clip.projector_type".into(), Value::String("ldp".into()));
    let err = VisionModel::from_mmproj(&ct, &mut f, &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(err.contains("qwen3vl_merger"), "{err}");
    ct.metadata.insert(
        "clip.projector_type".into(),
        Value::String("qwen3vl_merger".into()),
    );
    ct.metadata.insert(
        "clip.vision.is_deepstack_layers".into(),
        Value::Array(vec![Value::Bool(false), Value::Bool(true)]),
    );
    let err = VisionModel::from_mmproj(&ct, &mut f, &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(err.contains("deepstack"), "{err}");
    ct.metadata.remove("clip.vision.is_deepstack_layers");
    ct.metadata.remove("clip.vision.patch_size");
    let err = VisionModel::from_mmproj(&ct, &mut f, &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(err.contains("clip.vision.patch_size"), "{err}");
}
