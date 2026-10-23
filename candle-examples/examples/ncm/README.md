# candle-ncm: Next-Concept Modeling

[Next-Concept Modeling](https://arxiv.org/abs/2610.14162) (NCM), from Stanford's Hazy Research
and Together AI, generates text coarse to fine rather than token by token. A multiscale residual
VQ-VAE turns a sequence into a hierarchy of discrete codes, from a single code summarizing the
whole sequence up to one code per token. A transformer then predicts that hierarchy one level at
a time, with all the codes of a level predicted in parallel. Generating a 128-token story takes
11 forward passes, however long the text.

This example runs the released TinyStories model, for unconditional story generation. It covers
inference only, and the other released tasks are not supported yet: XSum and WMT14 need the
context encoder and classifier-free guidance.

## Converting the checkpoint

The checkpoints in
[hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints) are PyTorch
pickles, and the BPE vocabulary is a pickled Python dict. `convert.py` turns them into
safetensors and JSON once. It needs `torch`, `safetensors` and `pyyaml`.

```bash
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir ncm-checkpoints
curl -LO https://raw.githubusercontent.com/HazyResearch/ncm/main/config/eval/tinystories.yaml
python candle-examples/examples/ncm/convert.py \
    --checkpoint-dir ncm-checkpoints/tinystories --config tinystories.yaml --out ncm-tinystories
```

This writes `config.json`, `generator.safetensors`, `tokenizer.safetensors` and `vocab.json` to
`ncm-tinystories/`.

## Running the example

```bash
cargo run --example ncm --release -- --model-dir ncm-tinystories --n-samples 4
```

`--show-levels` also prints the text decoded after each level, so you can watch a story form from
a single code into full sentences. `--temperature`, `--top-k` and `--top-p` control how each
level's codes are sampled. The defaults match the reference evaluation config: temperature 1 with
no top-k or top-p filtering.

As in the reference evaluation, the first level is sampled from its unigram distribution over the
training set, which ships with the generator checkpoint. Every later level is predicted by the
model from the end-of-text prefix and the levels before it.
