//! Helpers shared by the tests that compare against the golden fixtures of `tests/fixtures`.
#![allow(dead_code)]

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn fixture(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(path)
}

pub fn load(path: &str) -> Result<HashMap<String, Tensor>> {
    candle::safetensors::load(fixture(path), &Device::Cpu)
}

pub fn read(path: &str) -> String {
    std::fs::read_to_string(fixture(path)).unwrap()
}

pub fn json<T: serde::de::DeserializeOwned>(path: &str) -> T {
    let mut v: serde_json::Value = serde_json::from_str(&read(path)).unwrap();
    // transformers 5 nests rope_theta under rope_parameters.
    if v.get("rope_theta").is_none() {
        if let Some(theta) = v
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .cloned()
        {
            v["rope_theta"] = theta;
        }
    }
    serde_json::from_value(v).unwrap()
}

pub fn vb(paths: &[&str]) -> Result<VarBuilder<'static>> {
    let files: Vec<PathBuf> = paths.iter().map(|p| fixture(p)).collect();
    unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu) }
}

/// The largest absolute difference, relative to the largest magnitude of `expected`.
pub fn rel_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    assert_eq!(actual.dims(), expected.dims(), "shape mismatch");
    let scale = expected
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?
        .max(1e-6);
    let d = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    Ok(d / scale)
}

pub fn assert_close(actual: &Tensor, expected: &Tensor, tol: f32, what: &str) {
    let d = rel_diff(actual, expected).unwrap();
    assert!(d <= tol, "{what}: relative difference {d:e} above {tol:e}");
}
