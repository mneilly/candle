#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::flux2::{self, model::Flux2};
use candle_transformers::models::{mistral, qwen3};
use clap::Parser;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(long, default_value = "A rusty robot walking on a beach")]
    prompt: String,

    #[arg(long, value_enum, default_value = "klein-base-9b")]
    model: Model,

    /// A Hugging Face repo to load the model from, in place of the default one of `--model`.
    #[arg(long)]
    model_id: Option<String>,

    /// A local directory with the checkpoint in the diffusers layout, in place of the hub.
    #[arg(long)]
    local_dir: Option<PathBuf>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image, a multiple of 16.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image, a multiple of 16.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// The number of denoising steps, 50 by default (4 for the distilled klein models).
    #[arg(long)]
    steps: Option<usize>,

    /// The guidance scale: embedded guidance for [dev], classifier-free guidance for the klein
    /// base models, ignored by the distilled klein models.
    #[arg(long, default_value_t = 4.0)]
    guidance: f64,

    /// The maximum number of prompt tokens; prompts are padded to it.
    #[arg(long, default_value_t = 512)]
    max_sequence_length: usize,

    /// The seed to use when generating the initial noise.
    #[arg(long)]
    seed: Option<u64>,

    /// The file to write the image to.
    #[arg(long, default_value = "flux2-output.png")]
    output: String,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// FLUX.2 [dev], with a Mistral 3 text encoder.
    Dev,
    /// FLUX.2 [klein] base 9B, with a Qwen3 text encoder and classifier-free guidance.
    #[value(name = "klein-base-9b")]
    KleinBase9b,
    /// FLUX.2 [klein] base 4B.
    #[value(name = "klein-base-4b")]
    KleinBase4b,
    /// FLUX.2 [klein] 9B, step-distilled.
    #[value(name = "klein-9b")]
    Klein9b,
    /// FLUX.2 [klein] 4B, step-distilled.
    #[value(name = "klein-4b")]
    Klein4b,
}

impl Model {
    fn repo(&self) -> &'static str {
        match self {
            Self::Dev => "black-forest-labs/FLUX.2-dev",
            Self::KleinBase9b => "black-forest-labs/FLUX.2-klein-base-9B",
            Self::KleinBase4b => "black-forest-labs/FLUX.2-klein-base-4B",
            Self::Klein9b => "black-forest-labs/FLUX.2-klein-9B",
            Self::Klein4b => "black-forest-labs/FLUX.2-klein-4B",
        }
    }

    fn is_dev(&self) -> bool {
        *self == Self::Dev
    }
}

/// The system prompt of the FLUX.2 [dev] text encoder.
const SYSTEM_MESSAGE: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation.";

/// Files of a checkpoint in the diffusers layout, from the hub or a local directory.
enum Source {
    Hub(candle_examples::hub::Repo<hf_hub::RepoTypeModel>),
    Local(PathBuf),
}

impl Source {
    fn get(&self, file: &str) -> Result<PathBuf> {
        match self {
            Self::Hub(repo) => Ok(repo.get(file)?),
            Self::Local(dir) => {
                let path = dir.join(file);
                if !path.exists() {
                    anyhow::bail!("{path:?} does not exist")
                }
                Ok(path)
            }
        }
    }

    fn json(&self, file: &str) -> Result<serde_json::Value> {
        Ok(serde_json::from_slice(&std::fs::read(self.get(file)?)?)?)
    }

    /// The safetensors files of a component: sharded with an index, or a single file.
    fn weights(&self, dir: &str, stem: &str) -> Result<Vec<PathBuf>> {
        match self.get(&format!("{dir}/{stem}.safetensors.index.json")) {
            Ok(index) => {
                let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
                let mut files: Vec<String> = index["weight_map"]
                    .as_object()
                    .ok_or_else(|| E::msg("no weight map in the index"))?
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                files.sort();
                files.dedup();
                files
                    .iter()
                    .map(|f| self.get(&format!("{dir}/{f}")))
                    .collect()
            }
            Err(_) => Ok(vec![self.get(&format!("{dir}/{stem}.safetensors"))?]),
        }
    }
}

/// transformers 5 nests `rope_theta` under `rope_parameters`; candle's configs read it at the
/// top level.
fn flatten_rope_theta(mut cfg: serde_json::Value) -> serde_json::Value {
    if cfg.get("rope_theta").is_none() {
        if let Some(theta) = cfg
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .cloned()
        {
            cfg["rope_theta"] = theta;
        }
    }
    cfg
}

/// Tokenizes `text`, truncated and right-padded to `max_len`, returning `(1, max_len)` ids and
/// the attention mask.
fn tokenize(
    tokenizer: &Tokenizer,
    text: &str,
    pad_id: u32,
    max_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut ids = tokenizer
        .encode(text, false)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    ids.truncate(max_len);
    let n = ids.len();
    ids.resize(max_len, pad_id);
    let mask: Vec<u32> = (0..max_len).map(|i| (i < n) as u32).collect();
    Ok((
        Tensor::new(ids, device)?.unsqueeze(0)?,
        Tensor::new(mask, device)?.unsqueeze(0)?,
    ))
}

/// The pad token of the tokenizer, from `tokenizer_config.json`.
fn pad_token_id(source: &Source, tokenizer: &Tokenizer, fallback: &str) -> Result<u32> {
    let pad = source
        .json("tokenizer/tokenizer_config.json")
        .ok()
        .and_then(|cfg| match &cfg["pad_token"] {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => o.get("content")?.as_str().map(String::from),
            _ => None,
        })
        .unwrap_or_else(|| fallback.to_string());
    tokenizer
        .token_to_id(&pad)
        .ok_or_else(|| E::msg(format!("the tokenizer has no pad token {pad}")))
}

/// Encodes prompts with the text encoder: the hidden states of three of its layers, stacked into
/// `(1, max_len, 3 * hidden)` embeddings.
struct TextEncoder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    pad_id: u32,
    max_len: usize,
    device: Device,
}

enum Encoder {
    Qwen3(qwen3::Model),
    Mistral(mistral::Model),
}

impl TextEncoder {
    fn load(
        source: &Source,
        dev: bool,
        max_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(source.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
        let cfg = source.json("text_encoder/config.json")?;
        let files = source.weights("text_encoder", "model")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        let (encoder, pad_fallback) = if dev {
            // Mistral3ForConditionalGeneration: the language model of a vision-language model.
            let text_cfg = flatten_rope_theta(cfg.get("text_config").cloned().unwrap_or(cfg));
            let cfg: mistral::Config = serde_json::from_value(text_cfg)?;
            let prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
                "model.language_model"
            } else {
                "language_model.model"
            };
            let model = mistral::Model::new_text_encoder(&cfg, vb.pp(prefix))?;
            (Encoder::Mistral(model), "<pad>")
        } else {
            let cfg: qwen3::Config = serde_json::from_value(flatten_rope_theta(cfg))?;
            (
                Encoder::Qwen3(qwen3::Model::new(&cfg, vb)?),
                "<|endoftext|>",
            )
        };
        let pad_id = pad_token_id(source, &tokenizer, pad_fallback)?;
        Ok(Self {
            encoder,
            tokenizer,
            pad_id,
            max_len,
            device: device.clone(),
        })
    }

    fn encode(&self, prompt: &str) -> Result<Tensor> {
        let (text, layers) = match &self.encoder {
            Encoder::Mistral(_) => (
                format!("<s>[SYSTEM_PROMPT]{SYSTEM_MESSAGE}[/SYSTEM_PROMPT][INST]{prompt}[/INST]"),
                [10, 20, 30],
            ),
            // The Qwen3 chat template with thinking disabled.
            Encoder::Qwen3(_) => (
                format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ),
                [9, 18, 27],
            ),
        };
        let (ids, mask) = tokenize(
            &self.tokenizer,
            &text,
            self.pad_id,
            self.max_len,
            &self.device,
        )?;
        let hidden = match &self.encoder {
            Encoder::Mistral(m) => m.forward_hidden_states(&ids, Some(&mask), &layers)?,
            Encoder::Qwen3(m) => m.forward_hidden_states(&ids, Some(&mask), &layers)?,
        };
        Ok(flux2::stack_text_embeddings(&hidden)?)
    }
}

/// Standard normal noise drawn on the host, so that a seed gives the same image on any device.
fn gaussian_noise(
    seed: Option<u64>,
    shape: (usize, usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    use rand::{Rng, SeedableRng};
    let mut rng = match seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    // Box-Muller.
    let values: Vec<f32> = (0..n)
        .map(|_| {
            let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.random();
            ((-2. * u1.ln()).sqrt() * (2. * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect();
    Ok(Tensor::from_vec(values, shape, device)?)
}

fn run(args: Args) -> Result<()> {
    if !args.height.is_multiple_of(16) || !args.width.is_multiple_of(16) {
        anyhow::bail!("the height and width must be multiples of 16")
    }
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let source = match &args.local_dir {
        Some(dir) => Source::Local(dir.clone()),
        None => {
            let api = candle_examples::hub::Api::new()?;
            Source::Hub(api.model(args.model_id.as_deref().unwrap_or(args.model.repo())))
        }
    };
    let dev = args.model.is_dev();
    let distilled = source
        .json("model_index.json")
        .ok()
        .and_then(|m| m["is_distilled"].as_bool())
        .unwrap_or(false);
    let cfg_scale = (!dev && !distilled && args.guidance > 1.).then_some(args.guidance);
    let steps = args.steps.unwrap_or(if distilled { 4 } else { 50 });

    // 1. Text embeddings, then free the text encoder.
    let (prompt_emb, negative_emb) = {
        let start = std::time::Instant::now();
        let te = TextEncoder::load(&source, dev, args.max_sequence_length, &device, dtype)?;
        let prompt = te.encode(&args.prompt)?;
        let negative = match cfg_scale {
            Some(_) => Some(te.encode("")?),
            None => None,
        };
        println!("encoded the prompt in {:?}", start.elapsed());
        (prompt, negative)
    };

    // 2. Denoising.
    let start = std::time::Instant::now();
    let cfg: flux2::model::Config =
        serde_json::from_value(source.json("transformer/config.json")?)?;
    let files = source.weights("transformer", "diffusion_pytorch_model")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let transformer = Flux2::new(&cfg, vb)?;
    let scheduler_cfg: flux2::SchedulerConfig = match source.json("scheduler/scheduler_config.json")
    {
        Ok(v) => serde_json::from_value(v)?,
        Err(_) => Default::default(),
    };
    println!("loaded the transformer in {:?}", start.elapsed());

    let (h, w) = (args.height / 16, args.width / 16);
    let img_ids = flux2::model::image_ids(h, w, 0, &device)?;
    let txt_ids = flux2::model::text_ids(prompt_emb.dim(1)?, &device)?;
    let scheduler = flux2::Scheduler::new(&scheduler_cfg, steps, h * w)?;
    let noise = gaussian_noise(args.seed, (1, cfg.in_channels, h, w), &device)?;
    let mut latents = flux2::pack(&noise)?.to_dtype(dtype)?;
    let guidance = if dev && cfg.guidance_embeds {
        Some(Tensor::new(&[args.guidance as f32], &device)?)
    } else {
        None
    };
    let start = std::time::Instant::now();
    for step in 0..scheduler.len() {
        let t = Tensor::new(&[scheduler.sigmas[step] as f32], &device)?;
        let mut v = transformer.forward(
            &latents,
            &img_ids,
            &prompt_emb,
            &txt_ids,
            &t,
            guidance.as_ref(),
        )?;
        if let (Some(scale), Some(neg)) = (cfg_scale, &negative_emb) {
            let u = transformer.forward(&latents, &img_ids, neg, &txt_ids, &t, None)?;
            v = flux2::apply_cfg(&v, &u, scale)?;
        }
        latents = scheduler.step(&latents, &v, step)?;
        println!("step {}/{} done, {:?}", step + 1, steps, start.elapsed());
    }
    drop(transformer);

    // 3. Decoding.
    let vae_cfg: flux2::VaeConfig = serde_json::from_value(source.json("vae/config.json")?)?;
    let files = source.weights("vae", "diffusion_pytorch_model")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let vae = flux2::Vae::new(&vae_cfg, vb)?;
    let image = vae.decode(&flux2::unpack(&latents, h, w)?)?;
    let image = flux2::to_u8_image(&image)?.squeeze(0)?;
    candle_examples::save_image(&image, &args.output)?;
    println!("saved {}", args.output);
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
