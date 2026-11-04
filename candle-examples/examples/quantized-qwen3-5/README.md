# candle-quantized-qwen3-5

Qwen3.5 and Qwen3.8 dense models from llama.cpp GGUF files (architecture `qwen35`), including
PrismML's [Ternary Bonsai](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) models.

Bonsai 2 27B is Qwen3.8-27B with ternary weights, `{-1, 0, 1}` times a scale per block of 128,
stored in the `PQ2_0` (2.125 bits per weight) or `PTQ1_0` (1.75 bits per weight) types of
PrismML's llama.cpp fork. The model may also fold Hadamard rotations into its weights
(`prism.hadamard.*` metadata) and carries MTP layers, with a trimmed draft head, for speculative
decoding.

```bash
$ cargo run --example quantized-qwen3-5 --release -- \
    --file <the .gguf file of the repository> --prompt "Explain the delta rule." --mtp-drafts 2
```

`--model` runs a local file instead. The tokenizer is read from the GGUF file. `--mtp-drafts N`
decodes greedily with the MTP layers drafting `N` tokens per step, which the model checks in one
pass; without it, `--temperature` and `--top-p` sample.

Images need the vision tower, which llama.cpp ships as a separate `mmproj` GGUF file: pass it
with `--mmproj` (a local path or a file of `--repo`) and each image with `--image`. The tower is
read only when there are images.

```bash
$ cargo run --example quantized-qwen3-5 --release -- --file <model .gguf> --mmproj <mmproj .gguf> \
    --image candle-examples/examples/yolo-v8/assets/bike.jpg --prompt "Describe this image."
```

The ternary types run on the CPU only for now. The context length of the file sizes the RoPE
tables; `--max-context` caps it.

The Gated DeltaNet layers normalize their queries and keys as ggml's `l2_norm`,
`x / max(|x|, eps)`, which PrismML's fork keeps for the models it packs. Files converted by
upstream llama.cpp expect `x / sqrt(|x|^2 + eps)`, as transformers: pass `--l2-norm sqrt`.
