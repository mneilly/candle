//! NCM against golden outputs of the reference implementation (HazyResearch/ncm).
//!
//! Variant `a` uses RoPE, tied codebook heads with a learned scale, QK-norm and the cascaded
//! learned level-0 upsampling; variant `b` learned positions, untied heads, a shared residual,
//! plain learned upsampling, a two-block decoder and sequence compression.

mod common;

use candle::{DType, Device, IndexOp, Tensor, D};
use candle_transformers::models::ncm::{Config, Model};
use common::{assert_close, json, load, vb};

fn model(variant: &str) -> (Model, Config) {
    let cfg: Config = json(&format!("ncm/{variant}/config.json"));
    let model = Model::new(
        &cfg,
        vb(&[&format!("ncm/{variant}/generator.safetensors")]).unwrap(),
        vb(&[&format!("ncm/{variant}/tokenizer.safetensors")]).unwrap(),
    )
    .unwrap();
    (model, cfg)
}

fn check_variant(variant: &str) {
    let (model, cfg) = model(variant);
    let r = load(&format!("ncm/{variant}/reference.safetensors")).unwrap();
    let l0 = r["l0"].to_dtype(DType::U32).unwrap();
    let prefix = r["prefix"].to_dtype(DType::U32).unwrap();
    let codes_ref = r["codes"].to_dtype(DType::U32).unwrap();
    let levels = cfg
        .vqvae
        .vector_quantizer_config
        .levels
        .manual_levels
        .clone();
    assert_eq!(model.levels(), levels.as_slice());

    // Greedy generation reproduces the reference codes exactly.
    let mut seen = vec![];
    let codes = model
        .generate(&prefix, &l0, |level, logits| {
            seen.push((level, logits.dims().to_vec()));
            logits.argmax(D::Minus1)
        })
        .unwrap();
    assert_eq!(
        codes.to_vec2::<u32>().unwrap(),
        codes_ref.to_vec2::<u32>().unwrap()
    );
    // The sampler sees every level after the first, with that level's length.
    let expected: Vec<(usize, Vec<usize>)> = (1..levels.len())
        .map(|l| (l, vec![3, levels[l], cfg.model.vocab_size[l - 1]]))
        .collect();
    assert_eq!(seen, expected);

    assert_close(
        &model.decode(&codes_ref).unwrap(),
        &r["logits"],
        1e-4,
        "decode",
    );
    for (i, l) in model.decode_levels(&codes_ref).unwrap().iter().enumerate() {
        assert_close(l, &r[&format!("level_logits.{i}")], 1e-4, "decode_levels");
    }
}

#[test]
fn variant_a_matches_reference() {
    check_variant("a")
}

#[test]
fn variant_b_matches_reference() {
    check_variant("b")
}

#[test]
fn reconstructions_add_up_level_contributions() {
    let (model, _) = model("a");
    let r = load("ncm/a/reference.safetensors").unwrap();
    let codes = r["codes"].to_dtype(DType::U32).unwrap();
    let q = model.quantizer();
    let recs = q.reconstructions(&codes).unwrap();
    assert_eq!(recs.len(), q.levels().len());
    let mut offset = 0;
    let mut sum: Option<Tensor> = None;
    for (level, &l) in q.levels().iter().enumerate() {
        let c = q
            .level_contribution(level, &codes.narrow(1, offset, l).unwrap())
            .unwrap();
        sum = Some(match sum {
            Some(s) => (s + c).unwrap(),
            None => c,
        });
        assert_close(&recs[level], sum.as_ref().unwrap(), 1e-6, "cumulative sum");
        offset += l;
    }
    // Downsampling to the last level is the identity, and to a coarser level averages.
    let full = recs.last().unwrap();
    let last = q.levels().len() - 1;
    assert_close(&q.downsample(full, last).unwrap(), full, 0., "identity");
    let coarse = q.downsample(full, 1).unwrap();
    assert_eq!(coarse.dim(1).unwrap(), q.levels()[1]);
    assert_close(
        &coarse.mean(1).unwrap(),
        &full.mean(1).unwrap(),
        1e-5,
        "area pooling keeps the mean",
    );
    // Codebook rows are the embeddings of their codes.
    let cb = q.codebook(2);
    let first = q
        .level_contribution(0, &codes.narrow(1, 0, 1).unwrap())
        .unwrap();
    assert_eq!(first.dims()[1], *q.levels().last().unwrap());
    assert_eq!(cb.dim(0).unwrap(), 32);
}

#[test]
fn decoder_matches_decode() {
    let (model, _) = model("b");
    let r = load("ncm/b/reference.safetensors").unwrap();
    let codes = r["codes"].to_dtype(DType::U32).unwrap();
    let z = model.quantizer().reconstructions(&codes).unwrap();
    let logits = candle::Module::forward(model.decoder(), z.last().unwrap()).unwrap();
    assert_close(&logits, &model.decode(&codes).unwrap(), 0., "decoder");
}

#[test]
fn padded_prefix_is_masked() {
    // A pad token at prefix position 1 must not influence the generated codes, whatever it
    // is replaced with, as long as it is the pad id.
    let (model, _) = model("a");
    let dev = Device::Cpu;
    let l0 = Tensor::new(&[[5u32]], &dev).unwrap();
    let greedy = |prefix: &Tensor| {
        model
            .generate(prefix, &l0, |_, logits| logits.argmax(D::Minus1))
            .unwrap()
    };
    let padded = greedy(&Tensor::new(&[[0u32, 1]], &dev).unwrap());
    let other = greedy(&Tensor::new(&[[0u32, 7]], &dev).unwrap());
    assert_eq!(padded.i(0).unwrap().dims(), other.i(0).unwrap().dims());
    assert_ne!(
        padded.to_vec2::<u32>().unwrap(),
        other.to_vec2::<u32>().unwrap(),
        "an unpadded token does change the generation"
    );
}

#[test]
fn rejects_unsupported_configs() {
    let base: Config = json("ncm/a/config.json");
    let gen = || vb(&["ncm/a/generator.safetensors"]).unwrap();
    let tok = || vb(&["ncm/a/tokenizer.safetensors"]).unwrap();
    let mut cfg = base.clone();
    cfg.model.query_level0 = true;
    let err = Model::new(&cfg, gen(), tok()).unwrap_err().to_string();
    assert!(err.contains("query_level0"), "{err}");
    let mut cfg = base.clone();
    cfg.model.vocab_size.pop();
    let err = Model::new(&cfg, gen(), tok()).unwrap_err().to_string();
    assert!(err.contains("generator heads"), "{err}");
}
