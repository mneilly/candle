#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::ncm::{Config, Model};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Directory holding the files written by `convert.py`: config.json, generator.safetensors,
    /// tokenizer.safetensors and vocab.json.
    #[arg(long)]
    model_dir: std::path::PathBuf,

    /// The number of stories to generate, as one batch.
    #[arg(long, default_value_t = 4)]
    n_samples: usize,

    /// The seed to use when sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The temperature used to sample each level's codes, 0 for greedy.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Only sample among the top K codes.
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Also print the text decoded after each level, to show it forming coarse to fine.
    #[arg(long)]
    show_levels: bool,
}

/// The byte-level BPE vocabulary of the reference tokenizer, used to turn ids back into text.
struct Vocab {
    tokens: Vec<Vec<u8>>,
    eos: u32,
    pad: u32,
}

impl Vocab {
    fn decode(&self, ids: &[u32]) -> String {
        let ids = match ids.iter().position(|&t| t == self.eos) {
            Some(end) => &ids[..end],
            None => ids,
        };
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&t| t != self.pad)
            .flat_map(|&t| self.tokens.get(t as usize).into_iter().flatten().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decodes `(batch, len, vocab)` logits greedily, one string per batch element.
    fn decode_logits(&self, logits: &Tensor) -> Result<Vec<String>> {
        let ids = logits.argmax(candle::D::Minus1)?.to_vec2::<u32>()?;
        Ok(ids.iter().map(|ids| self.decode(ids)).collect())
    }
}

/// Samples every position of a `(batch, len, vocab)` logits tensor independently.
fn sample_level(lp: &mut LogitsProcessor, logits: &Tensor, device: &Device) -> Result<Tensor> {
    let (b, l, v) = logits.dims3()?;
    let logits = logits
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .reshape((b * l, v))?;
    let codes = (0..b * l)
        .map(|i| lp.sample(&logits.get(i)?))
        .collect::<candle::Result<Vec<u32>>>()?;
    Ok(Tensor::from_vec(codes, (b, l), device)?)
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dir = &args.model_dir;
    let config: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let tokens: Vec<Vec<u8>> = serde_json::from_slice(&std::fs::read(dir.join("vocab.json"))?)?;
    let vocab = Vocab {
        tokens,
        eos: config.eos_token_id,
        pad: config.pad_token_id,
    };
    let generator_file = dir.join("generator.safetensors");
    let tokenizer_file = dir.join("tokenizer.safetensors");
    let dtype = DType::F32;
    let vb_generator =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&generator_file], dtype, &device)? };
    let vb_tokenizer =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&tokenizer_file], dtype, &device)? };
    let model = Model::new(&config, vb_generator, vb_tokenizer)?;
    // The released checkpoints sample the first level from its unigram distribution over the
    // training set rather than predicting it.
    let level0_unigram = unsafe { candle::safetensors::MmapedSafetensors::new(&generator_file)? }
        .load("level0_unigram", &Device::Cpu)
        .map_err(|_| E::msg("generator.safetensors has no level0_unigram, re-run convert.py"))?
        .to_dtype(DType::F32)?;
    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let n = args.n_samples;
    let levels = model.levels().to_vec();
    let mut l0_sampler =
        LogitsProcessor::from_sampling(args.seed, Sampling::All { temperature: 1. });
    let l0_logits = level0_unigram.log()?;
    let level0 = (0..n * levels[0])
        .map(|_| l0_sampler.sample(&l0_logits))
        .collect::<candle::Result<Vec<u32>>>()?;
    let level0 = Tensor::from_vec(level0, (n, levels[0]), &device)?;

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        let temperature = args.temperature;
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    let mut lp = LogitsProcessor::from_sampling(args.seed.wrapping_add(1), sampling);
    // Unconditional generation: the prefix is the end-of-text token.
    let prefix = Tensor::full(config.eos_token_id, (n, config.model.prefix_len), &device)?;
    let codes = model.generate(&prefix, &level0, |_, logits| {
        sample_level(&mut lp, logits, &device).map_err(candle::Error::wrap)
    })?;
    let texts = vocab.decode_logits(&model.decode(&codes)?)?;
    let dt = start.elapsed();

    if args.show_levels {
        let per_level = model
            .decode_levels(&codes)?
            .iter()
            .map(|logits| vocab.decode_logits(logits))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..n {
            println!("\n=== sample {i}: decoded after each level ===");
            for (level, texts) in per_level.iter().enumerate() {
                println!("[level {level:2}, {:3} codes] {}", levels[level], texts[i]);
            }
        }
    }
    for (i, text) in texts.iter().enumerate() {
        println!("\n=== sample {i} ===\n{text}");
    }
    println!(
        "\n{n} samples of {} tokens in {} forward passes, {dt:.2?}",
        levels.last().unwrap(),
        levels.len() - 1,
    );
    Ok(())
}
