#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3_5::{Config, L2Norm, Model};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum L2NormArg {
    /// ggml's `x / max(|x|, eps)`, which the PrismML (Ternary Bonsai) files expect.
    Max,
    /// `x / sqrt(|x|^2 + eps)`, for files converted by upstream llama.cpp.
    Sqrt,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// A local `qwen35` GGUF file. Otherwise `--file` is downloaded from `--repo`.
    #[arg(long)]
    model: Option<String>,

    /// The Hub repository of `--file`.
    #[arg(long, default_value = "prism-ml/Ternary-Bonsai-2-27B-gguf")]
    repo: String,

    /// The GGUF file of `--repo` to run.
    #[arg(long)]
    file: Option<String>,

    /// A `tokenizer.json`; by default the tokenizer stored in the GGUF file is used.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value = "Write a haiku about ternary weights.")]
    prompt: String,

    /// Skip the chat template.
    #[arg(long)]
    no_chat_template: bool,

    /// The number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 512)]
    sample_len: usize,

    /// The sampling temperature; 0 decodes greedily.
    #[arg(long, default_value_t = 0.)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Speculative greedy decoding: the MTP layers draft this many tokens per step.
    #[arg(long)]
    mtp_drafts: Option<usize>,

    /// How the Gated DeltaNet layers normalize queries and keys.
    #[arg(long, value_enum, default_value_t = L2NormArg::Max)]
    l2_norm: L2NormArg,

    /// Cap the context length, which sizes the RoPE tables.
    #[arg(long)]
    max_context: Option<usize>,

    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_path = match (&args.model, &args.file) {
        (Some(path), _) => std::path::PathBuf::from(path),
        (None, Some(file)) => candle_examples::hub::Api::new()?
            .model(args.repo.clone())
            .get(file)?,
        (None, None) => anyhow::bail!("pass --model, or --file of a GGUF in --repo"),
    };
    let device = candle_examples::device(args.cpu)?;
    if !device.is_cpu() {
        println!("the ternary types run on the CPU only for now");
    }

    let start = std::time::Instant::now();
    let mut file = std::fs::File::open(&model_path)?;
    let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
    let mut cfg = Config::from_gguf(&ct)?;
    cfg.l2_norm = match args.l2_norm {
        L2NormArg::Max => L2Norm::Max,
        L2NormArg::Sqrt => L2Norm::Sqrt,
    };
    if let Some(n) = args.max_context {
        cfg.max_position_embeddings = cfg.max_position_embeddings.min(n);
    }
    let tokenizer = match &args.tokenizer {
        Some(path) => Tokenizer::from_file(path).map_err(E::msg)?,
        None => Tokenizer::from_gguf(&ct)?,
    };
    let mut model = Model::new(&cfg, &ct, &mut file, &device)?;
    println!(
        "loaded {} layers and {} MTP layers{} in {:.1}s",
        cfg.num_hidden_layers,
        cfg.num_mtp_layers,
        if model.has_draft_head() {
            " with a draft head"
        } else {
            ""
        },
        start.elapsed().as_secs_f32()
    );

    let prompt = if args.no_chat_template {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    };
    let ids = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let ids = ids.get_ids().to_vec();
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    let mut tos = TokenOutputStream::new(tokenizer);
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let mut generated = 0usize;
    let start = std::time::Instant::now();

    match args.mtp_drafts {
        Some(drafts) => {
            if !model.has_mtp() {
                anyhow::bail!("the file has no MTP layers")
            }
            let stats = model.generate_mtp(&input, args.sample_len, drafts, |t| {
                generated += 1;
                if eos.contains(&t) {
                    return Ok(false);
                }
                if let Some(t) = tos.next_token(t).map_err(candle::Error::wrap)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                Ok(true)
            })?;
            println!(
                "\n{generated} tokens ({:.2} token/s), {} of {} drafts accepted in {} steps",
                generated as f64 / start.elapsed().as_secs_f64(),
                stats.accepted,
                stats.drafted,
                stats.rounds
            );
        }
        None => {
            let sampling = if args.temperature <= 0. {
                Sampling::ArgMax
            } else {
                match args.top_p {
                    Some(p) => Sampling::TopP {
                        p,
                        temperature: args.temperature,
                    },
                    None => Sampling::All {
                        temperature: args.temperature,
                    },
                }
            };
            let mut sampler = LogitsProcessor::from_sampling(args.seed, sampling);
            let mut logits = model.forward(&input, 0)?;
            for index in 0..args.sample_len {
                let t = sampler.sample(&logits.flatten_all()?)?;
                generated += 1;
                if eos.contains(&t) {
                    break;
                }
                if let Some(t) = tos.next_token(t)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                let next = Tensor::new(&[[t]], &Device::Cpu)?.to_device(&device)?;
                logits = model.forward(&next, ids.len() + index)?;
            }
            println!(
                "\n{generated} tokens ({:.2} token/s)",
                generated as f64 / start.elapsed().as_secs_f64()
            );
        }
    }
    if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    Ok(())
}
