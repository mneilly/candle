# candle-flux2: text-to-image with FLUX.2

FLUX.2 is the second generation of the FLUX rectified-flow transformers from Black Forest Labs.
Its text conditioning comes from the hidden states of three intermediate layers of a language
model: Mistral 3 for FLUX.2 [dev], Qwen3 for FLUX.2 [klein].

This example loads checkpoints in the diffusers layout, from the Hugging Face hub or a local
directory, and supports text-to-image generation.

## Running the example

```bash
cargo run --features cuda --example flux2 --release -- \
    --model klein-base-9b --prompt "A rusty robot walking on a beach"
```

`--model` selects the variant:

| `--model` | Repo | Text encoder | Guidance |
|---|---|---|---|
| `dev` | `black-forest-labs/FLUX.2-dev` | Mistral 3 | embedded, `--guidance` (default 4) |
| `klein-base-9b`, `klein-base-4b` | `black-forest-labs/FLUX.2-klein-base-{9B,4B}` | Qwen3 | classifier-free, `--guidance` |
| `klein-9b`, `klein-4b` | `black-forest-labs/FLUX.2-klein-{9B,4B}` | Qwen3 | none, step-distilled |

Whether a klein checkpoint is step-distilled is read from its `model_index.json`; the distilled
ones default to 4 steps, the others to 50 (`--steps`). `--model-id` loads another repo with the
same layout, and `--local-dir` a local copy.

The image size is set with `--height` and `--width`, multiples of 16 (1024x1024 by default), and
`--seed` makes the initial noise reproducible on any device.

## Not supported yet

- Reference images, for editing and multi-image conditioning.
- The prompt upsampling of the [dev] pipeline, which rewrites the prompt with the vision-language
  text encoder.
- Quantized weights. FLUX.2 [dev] is large: with the default config its transformer has about
  32B parameters, and its Mistral 3 text encoder is a 24B-class model.
