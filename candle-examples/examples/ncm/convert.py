# Converts a released NCM checkpoint to the files used by the candle example.
#
# The checkpoints in hazyresearch/ncm-checkpoints are PyTorch pickles and the BPE vocabulary is a
# pickled dict, so this script turns them into safetensors and JSON once:
#
#   hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir ncm-checkpoints
#   curl -LO https://raw.githubusercontent.com/HazyResearch/ncm/main/config/eval/tinystories.yaml
#   python convert.py --checkpoint-dir ncm-checkpoints/tinystories \
#       --config tinystories.yaml --out ncm-tinystories
#
# Requirements: torch, safetensors, pyyaml, plus omegaconf if the checkpoints pickle configs.
import argparse
import json
import os
import pickle

import torch
import yaml
from safetensors.torch import save_file

# Training-only buffers of the quantizer, not needed for inference.
SKIP_TOKENIZER = ("ema_cluster_size_", "ema_w_", "codebook_hit_", "_km_done", "_fd_step")


def tensors(state_dict, skip=()):
    out = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or any(s in name for s in skip):
            continue
        out[name] = value.detach().cpu().contiguous().clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="directory with generator.pt, tokenizer.pt and the BPE pickles")
    ap.add_argument("--config", required=True, help="the task's eval config, e.g. config/eval/tinystories.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt = cfg["checkpoints"]
    os.makedirs(args.out, exist_ok=True)

    gen = torch.load(os.path.join(args.checkpoint_dir, ckpt["generator"]), map_location="cpu", weights_only=False)
    gen_tensors = tensors(gen["model"])
    level0 = gen.get("inference", {}).get("level0_unigram")
    if level0 is not None:
        gen_tensors["level0_unigram"] = level0.detach().float().cpu().contiguous()
    save_file(gen_tensors, os.path.join(args.out, "generator.safetensors"))

    tok_path = os.path.join(args.checkpoint_dir, ckpt.get("tokenizer", "tokenizer.pt"))
    tok = torch.load(tok_path, map_location="cpu", weights_only=False)
    save_file(tensors(tok["model"], SKIP_TOKENIZER), os.path.join(args.out, "tokenizer.safetensors"))

    with open(os.path.join(args.checkpoint_dir, ckpt["bpe_vocab"]), "rb") as f:
        vocab = pickle.load(f)
    tokens = [[] for _ in range(max(vocab) + 1)]
    for idx, token in vocab.items():
        tokens[idx] = list(token)
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(tokens, f)

    # Special token ids follow the order of special_tokens in the config, as in BPETokenizer.
    special = list(cfg["tokenizer"]["pre_tokenizer"]["special_tokens"].values())
    config = {
        "model": cfg["model"],
        "vqvae": cfg["tokenizer"]["vqvae"],
        "eos_token_id": special.index("<|endoftext|>"),
        "pad_token_id": special.index("<|pad|>"),
        "generation": cfg.get("generation", {}),
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {args.out}/: generator.safetensors ({len(gen_tensors)} tensors), tokenizer.safetensors, vocab.json, config.json")


if __name__ == "__main__":
    main()
