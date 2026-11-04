//! The multimodal Qwen3.5 against transformers' `Qwen3_5ForConditionalGeneration` and its image
//! processor (`Qwen2VLImageProcessorPil`), with two images in the prompt.

mod common;

use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen3_5_vl::{
    rope_positions, smart_resize, Config, Grid, ImageProcessor, Images, Model,
};
use common::{assert_close, load, read, vb};

fn config() -> Config {
    Config::from_json(&read("qwen3_5/vision/config.json")).unwrap()
}

fn model(vision: bool) -> Model {
    Model::new(
        &config(),
        vb(&["qwen3_5/vision/model.safetensors"]).unwrap(),
        vision,
    )
    .unwrap()
}

/// The pixel values and grids of the two images of the reference.
fn images() -> (Tensor, Vec<Grid>) {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let processor = ImageProcessor::new(&config().vision);
    let (mut pixels, mut grids) = (vec![], vec![]);
    for i in 0..2 {
        let (p, g) = processor.preprocess(&r[&format!("image.{i}")]).unwrap();
        pixels.push(p);
        grids.push(g);
    }
    (Tensor::cat(&pixels, 0).unwrap(), grids)
}

#[test]
fn config_parsing() {
    let cfg = config();
    assert_eq!(
        (
            cfg.image_token_id,
            cfg.vision_start_token_id,
            cfg.vision_end_token_id
        ),
        (45, 46, 47)
    );
    assert_eq!((cfg.vision.depth, cfg.vision.patch_size), (2, 4));
    assert!(cfg.text.tie_word_embeddings);
    let mrope = cfg.text.rope_parameters.as_ref().unwrap();
    assert_eq!(mrope.mrope_section, Some(vec![2, 1, 1]));
    assert!(Config::from_json("{}").is_err());
}

#[test]
fn image_processor_matches() {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let (pixels, grids) = images();
    assert_close(&pixels, &r["pixel_values"], 1e-6, "pixel values");
    let expected: Vec<Vec<i64>> = r["image_grid_thw"].to_vec2().unwrap();
    let grids: Vec<Vec<i64>> = grids
        .iter()
        .map(|&(t, h, w)| vec![t as i64, h as i64, w as i64])
        .collect();
    assert_eq!(grids, expected);
    let processor = ImageProcessor::new(&config().vision);
    assert_eq!(processor.factor(), 8);
    let odd = Tensor::zeros((12, 8, 3), DType::U8, &Device::Cpu).unwrap();
    assert!(processor.preprocess(&odd).is_err());
}

#[test]
fn smart_resize_matches() {
    // From transformers' smart_resize: rounding, the pixel cap and the floor.
    for ((h, w, f, min, max), expected) in [
        ((100, 200, 28, 3136, 1003520), (112, 196)),
        ((1080, 1920, 32, 65536, 1048576), (768, 1344)),
        ((10, 10, 32, 65536, 16777216), (256, 256)),
        ((4000, 3000, 32, 4096, 262144), (576, 416)),
        ((42, 1050, 28, 3136, 1003520), (56, 1064)),
    ] {
        assert_eq!(smart_resize(h, w, f, min, max).unwrap(), expected);
    }
    assert!(smart_resize(1, 300, 28, 0, usize::MAX).is_err());
}

#[test]
fn vision_tower_matches() {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let m = model(true);
    let (pixels, grids) = images();
    let out = m.vision().unwrap().forward(&pixels, &grids).unwrap();
    assert_close(&out, &r["vision_out"], 1e-4, "vision");
    assert_eq!(m.vision().unwrap().config().out_hidden_size, 32);
}

#[test]
fn rope_positions_match() {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let ids: Vec<u32> = r["input_ids"]
        .to_dtype(DType::U32)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();
    let (_, grids) = images();
    let (pos, delta) = rope_positions(&ids, &grids, 45, 2).unwrap();
    let expected: Vec<Vec<i64>> = r["position_ids"].squeeze(1).unwrap().to_vec2().unwrap();
    for (axis, e) in expected.iter().enumerate() {
        let got: Vec<i64> = pos[axis].iter().map(|&p| p as i64).collect();
        assert_eq!(&got, e, "axis {axis}");
    }
    let d: Vec<Vec<i64>> = r["rope_delta"].to_vec2().unwrap();
    assert_eq!(delta, d[0][0]);
    // The image runs must match the grids.
    assert!(rope_positions(&ids, &grids[..1], 45, 2).is_err());
    assert!(rope_positions(&ids[..5], &grids, 45, 2).is_err());
    let extra = [&grids[..], &grids[..1]].concat();
    assert!(rope_positions(&ids, &extra, 45, 2).is_err());
}

#[test]
fn prompt_with_images_then_decode() {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let mut m = model(true);
    assert!(m.has_vision());
    let ids = r["input_ids"].to_dtype(DType::U32).unwrap();
    let (pixels, grids) = images();
    let images = Images {
        pixel_values: &pixels,
        grids: &grids,
    };
    let logits = m.forward_prompt(&ids, Some(images)).unwrap();
    assert_close(&logits, &r["prefill_logits"], 1e-4, "prefill");
    assert_eq!(m.text().rope_delta(), -3);
    // Decode steps take RoPE positions shifted by the delta.
    let l = ids.dim(1).unwrap();
    let toks = r["decode_tokens"].to_dtype(DType::U32).unwrap();
    for i in 0..toks.dim(1).unwrap() {
        let logits = m.forward(&toks.narrow(1, i, 1).unwrap(), l + i).unwrap();
        assert_close(&logits, &r[&format!("decode_logits.{i}")], 1e-4, "decode");
    }
    // Clearing the caches resets the delta.
    m.text_mut().clear_kv_cache();
    assert_eq!(m.text().rope_delta(), 0);
}

#[test]
fn vision_is_optional() {
    let r = load("qwen3_5/vision/reference.safetensors").unwrap();
    let ids = r["input_ids"].to_dtype(DType::U32).unwrap();
    let mut text_only = model(false);
    assert!(!text_only.has_vision() && text_only.vision().is_none());
    let (pixels, grids) = images();
    let images = Images {
        pixel_values: &pixels,
        grids: &grids,
    };
    let err = text_only
        .forward_prompt(&ids, Some(images))
        .unwrap_err()
        .to_string();
    assert!(err.contains("without its vision tower"), "{err}");
    // A text prompt gives the same logits with or without the tower.
    let a = text_only.forward_prompt(&ids, None).unwrap();
    let b = model(true).forward_prompt(&ids, None).unwrap();
    assert_close(&a, &b, 0., "text only");
    assert_eq!(text_only.config().image_token_id, 45);
}

/// The safetensors of the fixture, recording the tensors read.
struct Recording {
    tensors: std::collections::HashMap<String, Tensor>,
    read: std::sync::Mutex<Vec<String>>,
}

impl candle_nn::var_builder::SimpleBackend for Recording {
    fn get(
        &self,
        s: candle::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        assert_eq!(t.shape(), &s, "{name}");
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        self.read.lock().unwrap().push(name.to_string());
        self.tensors[name].to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[test]
fn text_only_never_reads_the_vision_weights() {
    for vision in [false, true] {
        let rec = std::sync::Arc::new(Recording {
            tensors: load("qwen3_5/vision/model.safetensors").unwrap(),
            read: Default::default(),
        });
        struct Shared(std::sync::Arc<Recording>);
        impl candle_nn::var_builder::SimpleBackend for Shared {
            fn get(
                &self,
                s: candle::Shape,
                name: &str,
                h: candle_nn::Init,
                dtype: DType,
                dev: &Device,
            ) -> candle::Result<Tensor> {
                self.0.get(s, name, h, dtype, dev)
            }
            fn get_unchecked(
                &self,
                name: &str,
                dtype: DType,
                dev: &Device,
            ) -> candle::Result<Tensor> {
                self.0.get_unchecked(name, dtype, dev)
            }
            fn contains_tensor(&self, name: &str) -> bool {
                self.0.contains_tensor(name)
            }
        }
        let vb = candle_nn::VarBuilder::from_backend(
            Box::new(Shared(rec.clone())),
            DType::F32,
            Device::Cpu,
        );
        Model::new(&config(), vb, vision).unwrap();
        let read = rec.read.lock().unwrap();
        let visual = read
            .iter()
            .filter(|n| n.starts_with("model.visual."))
            .count();
        if vision {
            assert_eq!(
                visual,
                rec.tensors
                    .keys()
                    .filter(|n| n.starts_with("model.visual."))
                    .count()
            );
        } else {
            assert_eq!(visual, 0);
            assert!(!read.is_empty());
        }
    }
}
