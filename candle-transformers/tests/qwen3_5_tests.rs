//! Qwen3.5 against golden outputs of transformers' `Qwen3_5ForCausalLM`.
//!
//! `text` is a text-only checkpoint. `multimodal` uses the multimodal layout: weights under
//! `model.language_model`, a nested text config with tied embeddings, layer types derived from
//! the full-attention interval, and the rope settings at the top level.

mod common;

use candle::DType;
use candle_transformers::models::qwen3_5::{Config, LayerType, Model};
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
