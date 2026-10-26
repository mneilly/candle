//! FLUX.2 against golden outputs of diffusers 0.42 and transformers 5.20: the text encoders, the
//! transformer, the VAE, the scheduler and the whole `Flux2KleinPipeline`.

mod common;

use candle::{DType, Device, Tensor};
use candle_transformers::models::flux2::{self, model::Flux2};
use candle_transformers::models::{mistral, qwen3};
use common::{assert_close, json, load, vb};
use std::collections::HashMap;

fn reference() -> HashMap<String, Tensor> {
    load("flux2/reference.safetensors").unwrap()
}

fn transformer(name: &str) -> Flux2 {
    let cfg: flux2::model::Config = json(&format!("flux2/{name}.json"));
    Flux2::new(&cfg, vb(&[&format!("flux2/{name}.safetensors")]).unwrap()).unwrap()
}

fn vae() -> flux2::Vae {
    let cfg: flux2::VaeConfig = json("flux2/vae.json");
    flux2::Vae::new(&cfg, vb(&["flux2/vae.safetensors"]).unwrap()).unwrap()
}

#[test]
fn text_encoder_hidden_states() {
    // Hidden states 1 and 2 are layer outputs, 3 (the layer count) the final-normed output.
    let r = reference();
    let ids = r["te_ids"].to_dtype(DType::U32).unwrap();
    let mask = r["te_mask"].to_dtype(DType::U32).unwrap();
    let qcfg: qwen3::Config = json("flux2/qwen3.json");
    let q = qwen3::Model::new(&qcfg, vb(&["flux2/qwen3.safetensors"]).unwrap()).unwrap();
    let mcfg: mistral::Config = json("flux2/mistral.json");
    let m = mistral::Model::new_text_encoder(&mcfg, vb(&["flux2/mistral.safetensors"]).unwrap())
        .unwrap();
    let qh = q
        .forward_hidden_states(&ids, Some(&mask), &[1, 2, 3])
        .unwrap();
    let mh = m
        .forward_hidden_states(&ids, Some(&mask), &[3, 1, 2])
        .unwrap();
    for (i, k) in [1, 2, 3].iter().enumerate() {
        assert_close(&qh[i], &r[&format!("qwen3_hidden.{k}")], 1e-5, "qwen3");
    }
    for (i, k) in [3, 1, 2].iter().enumerate() {
        assert_close(&mh[i], &r[&format!("mistral_hidden.{k}")], 1e-5, "mistral");
    }
    // Entry 0 is the input embeddings, and stacking interleaves the layers per token.
    let h0 = q.forward_hidden_states(&ids, None, &[0]).unwrap();
    assert_eq!(h0[0].dims(), &[2, 12, 16]);
    let stacked = flux2::stack_text_embeddings(&qh).unwrap();
    assert_eq!(stacked.dims(), &[2, 12, 48]);
    assert_close(&stacked.narrow(2, 16, 16).unwrap(), &qh[1], 0., "stacked");
    // A layer beyond the model is an error.
    let err = q
        .forward_hidden_states(&ids, None, &[4])
        .unwrap_err()
        .to_string();
    assert!(err.contains("3-layer"), "{err}");
    // An unpadded batch needs no mask: row 0 is unpadded.
    let row = q
        .forward_hidden_states(&ids.narrow(0, 0, 1).unwrap(), None, &[3])
        .unwrap();
    assert_close(
        &row[0],
        &r["qwen3_hidden.3"].narrow(0, 0, 1).unwrap(),
        1e-5,
        "unmasked",
    );
}

#[test]
fn transformer_matches_reference() {
    let r = reference();
    let dev = Device::Cpu;
    let img_ids = flux2::model::image_ids(4, 4, 0, &dev).unwrap();
    let txt_ids = flux2::model::text_ids(12, &dev).unwrap();
    let guidance = Tensor::new(&[3.5f32, 2.0], &dev).unwrap();
    for (name, g) in [
        ("transformer_guided", Some(&guidance)),
        ("transformer", None),
    ] {
        let t = transformer(name);
        assert_eq!(t.config().guidance_embeds, g.is_some());
        let out = t
            .forward(&r["img"], &img_ids, &r["txt"], &txt_ids, &r["timestep"], g)
            .unwrap();
        assert_close(&out, &r[&format!("{name}_out")], 1e-4, name);
    }
}

#[test]
fn vae_matches_reference() {
    let r = reference();
    let vae = vae();
    assert_close(
        &vae.decode(&r["vae_z"]).unwrap(),
        &r["vae_decoded"],
        1e-4,
        "decode",
    );
    assert_close(
        &vae.encode(&r["vae_pixels"]).unwrap(),
        &r["vae_encoded"],
        1e-4,
        "encode",
    );
}

#[test]
fn pipelines_match_reference() {
    // From prompt embeddings to the decoded image, step-distilled then with CFG.
    let r = reference();
    let dev = Device::Cpu;
    let scfg: flux2::SchedulerConfig = json("flux2/scheduler.json");
    let t = transformer("transformer");
    let vae = vae();
    let prompt = &r["pipe_prompt"];
    let txt_ids = flux2::model::text_ids(prompt.dim(1).unwrap(), &dev).unwrap();
    for (name, cfg_scale) in [("pipe_distilled", None), ("pipe_cfg", Some(3.0))] {
        let init = &r["pipe_latents"];
        let (_, _, h, w) = init.dims4().unwrap();
        let img_ids = flux2::model::image_ids(h, w, 0, &dev).unwrap();
        let sched = flux2::Scheduler::new(&scfg, 3, h * w).unwrap();
        let sigmas: Vec<f32> = sched.sigmas.iter().map(|&s| s as f32).collect();
        let sigmas = Tensor::new(sigmas, &dev).unwrap();
        assert_close(&sigmas, &r[&format!("{name}_sigmas")], 1e-6, "sigmas");
        let mut x = flux2::pack(init).unwrap();
        for step in 0..sched.len() {
            let ts = Tensor::new(&[sched.sigmas[step] as f32], &dev).unwrap();
            let mut v = t
                .forward(&x, &img_ids, prompt, &txt_ids, &ts, None)
                .unwrap();
            if let Some(s) = cfg_scale {
                let u = t
                    .forward(&x, &img_ids, &r["pipe_negative"], &txt_ids, &ts, None)
                    .unwrap();
                v = flux2::apply_cfg(&v, &u, s).unwrap();
            }
            x = sched.step(&x, &v, step).unwrap();
        }
        let image = vae.decode(&flux2::unpack(&x, h, w).unwrap()).unwrap();
        let image01 = ((image.clamp(-1f32, 1f32).unwrap() + 1.).unwrap() / 2.).unwrap();
        assert_close(&image01, &r[&format!("{name}_image")], 1e-4, name);
        // to_u8_image maps [-1, 1] to [0, 255].
        let u8_image = flux2::to_u8_image(&image).unwrap();
        let expected = (image01 * 255.).unwrap().round().unwrap();
        assert_close(
            &u8_image.to_dtype(DType::F32).unwrap(),
            &expected,
            1. / 255.,
            "u8",
        );
    }
}
