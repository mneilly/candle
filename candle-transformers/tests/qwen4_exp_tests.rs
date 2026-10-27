//! Qwen4-Exp (Qwen3.8-Flash-Next) against golden outputs of transformers' `Qwen4ExpForCausalLM`.
//!
//! The fixture has every component active: two gated residual streams, QSA on the
//! full-attention layer with a budget much smaller than the prompt, an n-gram embedding layer
//! stored as two shards, a sigmoid output gate on the Gated DeltaNet layers, per-expert MoE
//! weights, and end-of-text tokens in the prompt to exercise the n-gram window reset.

mod common;

use candle::safetensors::MmapedSafetensors;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen4_exp::{
    ngram_hash_constants, Config, LayerType, Model, PleTable, TokenIds,
};
use common::{assert_close, fixture, load, read, rel_diff, vb};
use std::collections::HashMap;
use std::sync::Arc;

fn config() -> Config {
    Config::from_json(&read("qwen4_exp/config.json")).unwrap()
}

fn reference() -> HashMap<String, Tensor> {
    load("qwen4_exp/reference.safetensors").unwrap()
}

/// Runs prefill, the cached decode steps and the continuation, returning the largest relative
/// difference from the reference.
fn max_diff(model: &mut Model) -> f32 {
    let r = reference();
    let u = |n: &str| r[n].to_dtype(DType::U32).unwrap();
    let prompt = u("prompt");
    let p = prompt.dim(1).unwrap();
    let mut worst = rel_diff(
        &model.forward_all(&prompt, 0).unwrap(),
        &r["prefill_logits"],
    )
    .unwrap();
    let toks = u("decode_tokens");
    for i in 0..toks.dim(1).unwrap() {
        let l = model
            .forward_all(&toks.narrow(1, i, 1).unwrap(), p + i)
            .unwrap();
        worst = worst.max(rel_diff(&l, &r[&format!("decode_logits.{i}")]).unwrap());
    }
    let cont = u("cont");
    let l = model.forward_all(&cont, p + toks.dim(1).unwrap()).unwrap();
    worst.max(rel_diff(&l, &r["cont_logits"]).unwrap())
}

#[test]
fn matches_reference_with_table_loaded() {
    let mut model = Model::new(&config(), vb(&["qwen4_exp/model.safetensors"]).unwrap()).unwrap();
    let d = max_diff(&mut model);
    assert!(d < 1e-4, "relative difference {d:e}");

    // After clearing the caches, forward gives the last position of the prefill.
    model.clear_kv_cache();
    let r = reference();
    let prompt = r["prompt"].to_dtype(DType::U32).unwrap();
    let p = prompt.dim(1).unwrap();
    let last = model.forward(&prompt, 0).unwrap();
    assert_close(
        &last,
        &r["prefill_logits"].narrow(1, p - 1, 1).unwrap(),
        1e-4,
        "forward",
    );

    // Clearing drops every cache, the QSA keys included: on a different sequence the cleared
    // model matches a fresh one.
    model.clear_kv_cache();
    let cont = r["cont"].to_dtype(DType::U32).unwrap();
    let mut fresh = Model::new(&config(), vb(&["qwen4_exp/model.safetensors"]).unwrap()).unwrap();
    assert_close(
        &model.forward_all(&cont, 0).unwrap(),
        &fresh.forward_all(&cont, 0).unwrap(),
        0.,
        "cleared",
    );
}

#[test]
fn matches_reference_with_memory_mapped_table() {
    let st = Arc::new(unsafe {
        MmapedSafetensors::new(fixture("qwen4_exp/model.safetensors")).unwrap()
    });
    let vb_all = vb(&["qwen4_exp/model.safetensors"]).unwrap();
    let prefix = Model::weight_prefix(&vb_all);
    assert_eq!(prefix, "model");
    let table = PleTable::from_mmaped(st.clone(), prefix, 1).unwrap();
    assert!(format!("{table:?}").contains("shards: 2"));
    let mut model = Model::new_with_ple_tables(&config(), vb_all, vec![table]).unwrap();
    let d = max_diff(&mut model);
    assert!(d < 1e-4, "relative difference {d:e}");
    // Layer 0 has no n-gram embedding table.
    let err = PleTable::from_mmaped(st, prefix, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no n-gram embedding table"), "{err}");
}

#[test]
fn matches_reference_with_fused_experts() {
    // Fuse the per-expert weights into the transformers layout: gate_up_proj (E, 2I, H) and
    // down_proj (E, H, I).
    let cfg = config();
    let mut w = load("qwen4_exp/model.safetensors").unwrap();
    for layer in 0..cfg.num_hidden_layers {
        let pre = format!("model.layers.{layer}.mlp.experts");
        let mut gate_up = vec![];
        let mut down = vec![];
        for e in 0..cfg.num_experts {
            let g = w.remove(&format!("{pre}.{e}.gate_proj.weight")).unwrap();
            let u = w.remove(&format!("{pre}.{e}.up_proj.weight")).unwrap();
            gate_up.push(Tensor::cat(&[g, u], 0).unwrap());
            down.push(w.remove(&format!("{pre}.{e}.down_proj.weight")).unwrap());
        }
        w.insert(
            format!("{pre}.gate_up_proj"),
            Tensor::stack(&gate_up, 0).unwrap(),
        );
        w.insert(format!("{pre}.down_proj"), Tensor::stack(&down, 0).unwrap());
    }
    let vb = VarBuilder::from_tensors(w, DType::F32, &Device::Cpu);
    let mut model = Model::new(&cfg, vb).unwrap();
    let d = max_diff(&mut model);
    assert!(d < 1e-4, "relative difference {d:e}");
}

#[test]
fn qsa_and_ngram_layers_matter() {
    // Without the QSA indexer, attention is dense and the logits move away from the reference.
    let mut cfg = config();
    cfg.indexer_n_heads = None;
    cfg.indexer_kv_heads = None;
    cfg.indexer_head_dim = None;
    cfg.indexer_budget = None;
    cfg.indexer_compress_ratio = None;
    let mut dense = Model::new(&cfg, vb(&["qwen4_exp/model.safetensors"]).unwrap()).unwrap();
    assert!(max_diff(&mut dense) > 1e-3);
}

#[test]
fn ple_layer_ids_are_normalized() {
    // Unsorted, duplicated ids are sorted and deduplicated as the reference does on load.
    let mut cfg = config();
    cfg.ple_layer_ids = vec![2, 2];
    assert_eq!(cfg.ple_layers(), vec![2]);
    let mut model = Model::new(&cfg, vb(&["qwen4_exp/model.safetensors"]).unwrap()).unwrap();
    assert!(max_diff(&mut model) < 1e-4);
}

#[test]
fn hash_constants_match_the_checkpoint() {
    let w = load("qwen4_exp/model.safetensors").unwrap();
    let ck = |n: &str| -> Vec<u64> {
        w[&format!("model.layers.1.ple.ple_embedding.{n}")]
            .to_vec1::<i64>()
            .unwrap()
            .iter()
            .map(|&v| v as u64)
            .collect()
    };
    let (m, v, o) = ngram_hash_constants(&config(), 0);
    assert_eq!(m, ck("layer_multipliers"));
    assert_eq!(v, ck("ngram_heads_vocab_sizes"));
    assert_eq!(o, ck("ngram_heads_offsets"));
}

#[test]
fn config_parsing() {
    let cfg = config();
    assert_eq!(cfg.eos_token_id, Some(TokenIds::Multiple(vec![3, 5])));
    assert_eq!(TokenIds::Multiple(vec![3, 5]).first(), Some(3));
    assert_eq!(TokenIds::Single(7).first(), Some(7));
    assert_eq!(TokenIds::Multiple(vec![]).first(), None);
    assert_eq!(
        cfg.layer_types(),
        [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::IndexedAttention
        ]
    );
    let mut derived = cfg.clone();
    derived.layer_types = None;
    derived.full_attention_interval = Some(2);
    assert_eq!(derived.layer_types()[1], LayerType::IndexedAttention);

    // The multimodal layout nests the text config; the top level fills in tie and eos.
    let mut text: serde_json::Value = serde_json::from_str(&read("qwen4_exp/config.json")).unwrap();
    text.as_object_mut().unwrap().remove("eos_token_id");
    let mm =
        serde_json::json!({"text_config": text, "tie_word_embeddings": true, "eos_token_id": 9});
    let cfg = Config::from_json(&mm.to_string()).unwrap();
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.eos_token_id, Some(TokenIds::Single(9)));
    assert!(Config::from_json("[]").is_err());
    assert!(Config::from_json(r#"{"text_config": 1}"#).is_err());
}

fn load_err(cfg: &Config) -> String {
    Model::new(cfg, vb(&["qwen4_exp/model.safetensors"]).unwrap())
        .unwrap_err()
        .to_string()
}

#[test]
fn rejects_invalid_configs() {
    let mut cfg = config();
    cfg.indexer_budget = None;
    assert!(load_err(&cfg).contains("incomplete"));

    let mut cfg = config();
    cfg.indexer_kv_heads = Some(2);
    assert!(load_err(&cfg).contains("indexer_kv_heads"));

    let mut cfg = config();
    cfg.indexer_budget = Some(6);
    assert!(load_err(&cfg).contains("multiple"));

    let mut cfg = config();
    cfg.eos_token_id = None;
    assert!(load_err(&cfg).contains("eos_token_id"));

    let mut cfg = config();
    cfg.ple_embed_dim = Some(30);
    assert!(load_err(&cfg).contains("ple_embed_dim"));

    let mut cfg = config();
    cfg.layer_types = Some(vec![LayerType::LinearAttention]);
    assert!(load_err(&cfg).contains("layer types"));
}

#[test]
fn config_defaults() {
    let text = r#"{"vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 4, "num_attention_heads": 1,
        "num_key_value_heads": 1, "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
        "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_num_key_heads": 1,
        "linear_num_value_heads": 1, "moe_intermediate_size": 4, "shared_expert_intermediate_size": 4,
        "num_experts_per_tok": 1, "num_experts": 2}"#;
    let cfg = Config::from_json(text).unwrap();
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.hidden_act, candle_nn::Activation::Silu);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert!(cfg.norm_topk_prob);
    assert_eq!((cfg.hc_count, cfg.hc_lowrank), (4, 320));
    assert_eq!(
        (
            cfg.ngram_size,
            cfg.heads_per_ngram,
            cfg.ple_conv_kernel_size
        ),
        (3, 8, 4)
    );
    assert_eq!(cfg.ngram_vocab_size_base, 20_000_000);
    assert_eq!(cfg.make_ngram_vocab_size_divisible_by, 128);
    assert_eq!(cfg.seed, 1234);
    assert!(cfg.ple_layer_ids.is_empty() && cfg.output_gate_type.is_none());
    // Without a nested eos or tie flag, the multimodal config keeps the text config's values.
    let mm = Config::from_json(&format!(r#"{{"text_config": {text}}}"#)).unwrap();
    assert_eq!(mm, cfg);
}

#[test]
fn variants_without_reference_run() {
    // Untopk-normalized routing, tied embeddings and a prompt too short for QSA to drop blocks.
    let mut cfg = config();
    cfg.norm_topk_prob = false;
    cfg.tie_word_embeddings = true;
    let mut model = Model::new(&cfg, vb(&["qwen4_exp/model.safetensors"]).unwrap()).unwrap();
    let ids = Tensor::new(&[[1u32, 2, 3, 4, 5, 6]], &Device::Cpu).unwrap();
    let logits = model.forward_all(&ids, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 6, 50]);
    let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn single_table_in_the_multimodal_layout() {
    // The multimodal layout with the n-gram table as a single tensor: loaded through the var
    // builder, and memory-mapped.
    let cfg = config();
    let mut w = load("qwen4_exp/model.safetensors").unwrap();
    let base = "model.layers.1.ple.ple_embedding.ngram_embedding";
    let shards: Vec<Tensor> = (0..2)
        .map(|i| w.remove(&format!("{base}.shard_{i}.weight")).unwrap())
        .collect();
    w.insert(format!("{base}.weight"), Tensor::cat(&shards, 0).unwrap());
    let w: HashMap<String, Tensor> = w
        .into_iter()
        .map(|(k, v)| (k.replacen("model.", "model.language_model.", 1), v))
        .collect();
    let path = std::env::temp_dir().join(format!(
        "qwen4_exp_single_{}.safetensors",
        std::process::id()
    ));
    candle::safetensors::save(&w, &path).unwrap();

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu).unwrap() };
    assert_eq!(Model::weight_prefix(&vb), "model.language_model");
    let mut model = Model::new(&cfg, vb.clone()).unwrap();
    assert!(max_diff(&mut model) < 1e-4);

    let st = Arc::new(unsafe { MmapedSafetensors::new(&path).unwrap() });
    let table = PleTable::from_mmaped(st, "model.language_model", 1).unwrap();
    assert!(format!("{table:?}").contains("shards: 1"));
    let mut model = Model::new_with_ple_tables(&cfg, vb, vec![table]).unwrap();
    assert!(max_diff(&mut model) < 1e-4);
    std::fs::remove_file(&path).unwrap();

    let t = PleTable::Tensor(Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap());
    assert!(format!("{t:?}").starts_with("Tensor"));
}

#[test]
fn missing_table_is_an_error() {
    let mut w = load("qwen4_exp/model.safetensors").unwrap();
    w.retain(|k, _| !k.contains("ngram_embedding"));
    let vb = VarBuilder::from_tensors(w, DType::F32, &Device::Cpu);
    let err = Model::new(&config(), vb).unwrap_err().to_string();
    assert!(err.contains("no n-gram embedding table"), "{err}");
}
