//! Qwen3.5 against golden outputs of transformers' `Qwen3_5ForCausalLM`.
//!
//! `text` is a text-only checkpoint. `multimodal` uses the multimodal layout: weights under
//! `model.language_model`, a nested text config with tied embeddings, layer types derived from
//! the full-attention interval, and the rope settings at the top level.

mod common;

use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen3_5::{Config, LayerType, Model, MtpStats};
use common::{assert_close, load, read, vb};

fn model(variant: &str) -> (Model, Config) {
    let cfg = Config::from_json(&read(&format!("qwen3_5/{variant}/config.json"))).unwrap();
    let model = Model::new(
        &cfg,
        vb(&[&format!("qwen3_5/{variant}/model.safetensors")]).unwrap(),
    )
    .unwrap();
    (model, cfg)
}

/// Prefill over two chunks, cached decode steps, then a continuation from the cached state.
fn check_variant(variant: &str) {
    let (mut model, _) = model(variant);
    let r = load(&format!("qwen3_5/{variant}/reference.safetensors")).unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let p = prompt.dim(1).unwrap();
    assert_close(
        &model.forward_all(&prompt, 0).unwrap(),
        &r["prefill_logits"],
        1e-4,
        "prefill",
    );
    let toks = r["decode_tokens"].to_dtype(DType::U32).unwrap();
    for i in 0..toks.dim(1).unwrap() {
        let l = model
            .forward_all(&toks.narrow(1, i, 1).unwrap(), p + i)
            .unwrap();
        assert_close(&l, &r[&format!("decode_logits.{i}")], 1e-4, "decode");
    }
    let cont = r["cont"].to_dtype(DType::U32).unwrap();
    let l = model.forward_all(&cont, p + toks.dim(1).unwrap()).unwrap();
    assert_close(&l, &r["cont_logits"], 1e-4, "continuation");

    // After clearing the caches the prompt gives the same logits again, and forward returns
    // the last position only.
    model.clear_kv_cache();
    let last = model.forward(&prompt, 0).unwrap();
    let expected = r["prefill_logits"].narrow(1, p - 1, 1).unwrap();
    assert_close(&last, &expected, 1e-4, "forward after clear_kv_cache");
}

#[test]
fn text_matches_reference() {
    check_variant("text")
}

#[test]
fn multimodal_layout_matches_reference() {
    check_variant("multimodal")
}

#[test]
fn config_parsing() {
    let text = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    let mm = Config::from_json(&read("qwen3_5/multimodal/config.json")).unwrap();
    assert!(!text.tie_word_embeddings);
    assert!(mm.tie_word_embeddings, "the top-level tie flag wins");
    assert!(mm.layer_types.is_none());
    let types = [
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::FullAttention,
    ];
    assert_eq!(text.layer_types(), types);
    assert_eq!(mm.layer_types(), types);
    let mut every_other = mm.clone();
    every_other.full_attention_interval = Some(2);
    assert_eq!(
        every_other.layer_types()[..2],
        [LayerType::LinearAttention, LayerType::FullAttention]
    );
    assert!(Config::from_json("{").is_err());
    assert!(Config::from_json(r#"{"text_config": {}}"#).is_err());
}

#[test]
fn layer_type_count_must_match() {
    let mut cfg = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    cfg.layer_types = Some(vec![LayerType::LinearAttention]);
    let err = Model::new(&cfg, vb(&["qwen3_5/text/model.safetensors"]).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("layer types"), "{err}");
}

#[test]
fn batch_rows_are_independent() {
    // Each row of a batch gives the same logits as on its own.
    let (mut model, _) = model("text");
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let both = model.forward_all(&prompt, 0).unwrap();
    model.clear_kv_cache();
    let row = model
        .forward_all(&prompt.narrow(0, 1, 1).unwrap(), 0)
        .unwrap();
    assert_close(&row, &both.narrow(0, 1, 1).unwrap(), 1e-5, "batch row");
}

#[test]
fn config_defaults() {
    // The optional fields of a minimal text config, and a multimodal config without a top-level
    // tie flag.
    let text = r#"{"vocab_size": 8, "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 4,
        "num_attention_heads": 1, "num_key_value_heads": 1, "max_position_embeddings": 16,
        "rms_norm_eps": 1e-6, "linear_key_head_dim": 2, "linear_value_head_dim": 2,
        "linear_num_key_heads": 1, "linear_num_value_heads": 1}"#;
    let cfg = Config::from_json(text).unwrap();
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.hidden_act, candle_nn::Activation::Silu);
    assert!(!cfg.tie_word_embeddings && !cfg.attention_bias);
    let mm = Config::from_json(&format!(r#"{{"text_config": {text}}}"#)).unwrap();
    assert_eq!(mm, cfg);
}

/// The MTP head over the prompt, teacher-forced with the main model's hidden states, then
/// chained drafts through its cache.
#[test]
fn mtp_matches_reference() {
    let (mut model, cfg) = model("text");
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert!(model.has_mtp());
    let r = load("qwen3_5/text/mtp_reference.safetensors").unwrap();
    let prompt = load("qwen3_5/text/reference.safetensors").unwrap()["prompt"]
        .to_dtype(DType::U32)
        .unwrap();
    let p = prompt.dim(1).unwrap();
    let hidden = model.forward_hidden(&prompt, 0).unwrap();
    assert_close(&hidden, &r["main_hidden"], 1e-4, "main hidden");
    let g = model
        .mtp_forward_hidden(
            &prompt.narrow(1, 1, p - 1).unwrap(),
            &hidden.narrow(1, 0, p - 1).unwrap(),
            0,
            0,
        )
        .unwrap();
    assert_close(&g, &r["teacher_hidden"], 1e-4, "teacher hidden");
    assert_close(
        &model.logits(&g).unwrap(),
        &r["teacher_logits"],
        1e-4,
        "teacher logits",
    );
    let inputs = r["draft_inputs"].to_dtype(DType::U32).unwrap();
    let mut h = hidden.narrow(1, p - 1, 1).unwrap();
    for i in 0..inputs.dim(1).unwrap() {
        let ids = inputs.narrow(1, i, 1).unwrap();
        h = model.mtp_forward_hidden(&ids, &h, p - 1 + i, i).unwrap();
        let logits = model.logits(&h).unwrap();
        assert_close(&logits, &r[&format!("draft_logits.{i}")], 1e-4, "draft");
    }
}

fn greedy(model: &mut Model, prompt: &Tensor, n: usize) -> Vec<u32> {
    model.clear_kv_cache();
    let mut tokens = vec![];
    let mut logits = model.forward(prompt, 0).unwrap();
    for i in 0..n {
        let t = logits.flatten_all().unwrap().argmax(0).unwrap();
        tokens.push(t.to_scalar::<u32>().unwrap());
        let ids = t.reshape((1, 1)).unwrap();
        logits = model.forward(&ids, prompt.dim(1).unwrap() + i).unwrap();
    }
    tokens
}

fn mtp_tokens(model: &mut Model, prompt: &Tensor, n: usize, drafts: usize) -> (Vec<u32>, MtpStats) {
    let mut tokens = vec![];
    let stats = model
        .generate_mtp(prompt, n, drafts, |t| {
            tokens.push(t);
            Ok(true)
        })
        .unwrap();
    (tokens, stats)
}

#[test]
fn generate_mtp_matches_greedy() {
    let (mut model, _) = model("text");
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    for row in 0..2 {
        let prompt = prompt.narrow(0, row, 1).unwrap();
        let expected = greedy(&mut model, &prompt, 24);
        for drafts in 1..=3 {
            let (tokens, stats) = mtp_tokens(&mut model, &prompt, 24, drafts);
            assert_eq!(tokens, expected, "row {row}, {drafts} drafts");
            assert_eq!(stats.drafted, stats.rounds * drafts);
            assert!(stats.accepted <= stats.drafted);
            // The prompt pass gives one token, each round its accepted drafts and one more.
            assert!(1 + stats.rounds + stats.accepted >= 24);
            assert!(1 + (stats.rounds - 1) + stats.accepted < 24);
        }
    }
}

#[test]
fn generate_mtp_stops() {
    let (mut model, _) = model("text");
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let prompt = prompt.narrow(0, 0, 1).unwrap();
    let expected = greedy(&mut model, &prompt, 5);
    for n in 0..=5 {
        let (tokens, _) = mtp_tokens(&mut model, &prompt, n, 2);
        assert_eq!(tokens, expected[..n], "max_new {n}");
    }
    // on_token stops the generation after the token it returns false on.
    for stop in 1..=5 {
        let mut tokens = vec![];
        model
            .generate_mtp(&prompt, 10, 3, |t| {
                tokens.push(t);
                Ok(tokens.len() < stop)
            })
            .unwrap();
        assert_eq!(tokens, expected[..stop], "stop after {stop}");
    }
    let err = model
        .generate_mtp(&prompt, 4, 0, |_| Ok(true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("at least one draft"), "{err}");
}

#[test]
fn mtp_is_optional() {
    // The multimodal fixture has no MTP weights, and a zero layer count skips them.
    let (mut mm, _) = model("multimodal");
    assert!(!mm.has_mtp());
    let ids = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
    let h = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
    let err = mm
        .mtp_forward_hidden(&ids, &h, 0, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no MTP head"), "{err}");
    let mut cfg = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    cfg.mtp_num_hidden_layers = 0;
    let text = Model::new(&cfg, vb(&["qwen3_5/text/model.safetensors"]).unwrap()).unwrap();
    assert!(!text.has_mtp());
}

#[test]
fn cache_snapshots_roll_back() {
    let (mut model, _) = model("text");
    let r = load("qwen3_5/text/reference.safetensors").unwrap();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let cont = r["cont"].to_dtype(DType::U32).unwrap();
    let p = prompt.dim(1).unwrap();
    model.forward(&prompt, 0).unwrap();
    let snapshot = model.cache();
    let first = model.forward_all(&cont, p).unwrap();
    model.forward_all(&cont, p + cont.dim(1).unwrap()).unwrap();
    model.set_cache(snapshot).unwrap();
    assert_close(
        &model.forward_all(&cont, p).unwrap(),
        &first,
        0.,
        "rolled back",
    );
    // A snapshot of another model does not fit.
    let mut cfg = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    cfg.mtp_num_hidden_layers = 0;
    let mm = Model::new(&cfg, vb(&["qwen3_5/text/model.safetensors"]).unwrap()).unwrap();
    let err = model.set_cache(mm.cache()).unwrap_err().to_string();
    assert!(err.contains("another model"), "{err}");
    // Nor does one with the layer types in another order.
    let mut cfg = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    let mut types = cfg.layer_types();
    types.reverse();
    cfg.layer_types = Some(types);
    let zeros = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
    let other = Model::new(&cfg, zeros).unwrap();
    let err = model.set_cache(other.cache()).unwrap_err().to_string();
    assert!(err.contains("another layer type"), "{err}");
}

#[test]
fn generate_mtp_accepts_every_matching_draft() {
    // With zero weights every logit is zero, so the model and its MTP head both pick token 0
    // and every draft is accepted.
    let cfg = Config::from_json(&read("qwen3_5/text/config.json")).unwrap();
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model = Model::new(&cfg, vb).unwrap();
    assert!(model.has_mtp());
    let prompt = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap();
    for drafts in 1..=3 {
        let (tokens, stats) = mtp_tokens(&mut model, &prompt, 13, drafts);
        assert_eq!(tokens, vec![0; 13]);
        assert_eq!(stats.accepted, stats.drafted, "{drafts} drafts");
        assert_eq!(stats.rounds, 12usize.div_ceil(drafts + 1));
    }
}
