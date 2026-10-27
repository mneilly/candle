# Generates the golden fixtures of the NCM, Qwen3.5, Qwen4-Exp and FLUX.2 tests.
#
# Each fixture is a tiny randomly initialised model built with the reference implementation,
# saved as safetensors, together with the reference outputs the candle tests compare against.
#
# Requirements: torch, safetensors, transformers==5.20.2, diffusers==0.42.0, omegaconf, hydra-core,
# and a checkout of https://github.com/HazyResearch/ncm for the NCM reference.
#
#   python generate.py --ncm-src /path/to/ncm --out candle-transformers/tests/fixtures
#
# Without --ncm-src only the Qwen and FLUX.2 fixtures are regenerated.
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def noisy(module, scale):
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.add_(scale * torch.randn_like(p))
    return module.eval()


def tensors(sd, skip=()):
    return {k: v.detach().contiguous().clone() for k, v in sd.items()
            if isinstance(v, torch.Tensor) and not any(s in k for s in skip)}


def save_ref(path, ref):
    save_file({k: (v.float() if v.is_floating_point() else v).contiguous().clone() for k, v in ref.items()}, path)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------------------------
# NCM
# ---------------------------------------------------------------------------------------------

def ncm_fixtures(out, ncm_src):
    sys.path.insert(0, ncm_src)
    from omegaconf import OmegaConf
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    levels = [1, 4, 8, 16]
    base_model = {
        "n_layer": 2, "n_head": 2, "bias": True, "dropout": 0.0, "n_embd": 32, "prefix_len": 2,
        "raw_token_prefix": True, "prefix_pad_token_id": 1, "vocab_size": [16] * 3,
        "predict_level0": False, "attn_pattern": "block_diagonal", "use_level_encoding": True,
        "use_rope": True, "rope_base": 100000, "use_qk_norm": True, "query_level0": False,
        "tie_heads_to_codebook": True, "tie_heads_learn_scale": True,
    }
    base_vq = {
        "n_layers": 2, "embed_dim": 16, "in_vocab_size": 30, "context_length": 16, "compression_factor": 1,
        "dropout": 0.0, "pre_quant_groupnorm": 4, "use_qk_norm": True, "vqvae_vocab_size": [16] * 4,
        "vector_quantizer_config": {
            "clss": "multiscale_residual_vector_quantizer", "decay": 0.95,
            "learned_l1_sampling": True, "learned_l1_cascade": True, "learned_l1_cascade_base": 4,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
            "levels": {"use_manual_levels": True, "manual_levels": levels},
        },
    }
    variants = {
        # RoPE, tied heads with a learned scale, QK-norm, cascaded learned level-0 upsampling.
        "a": ({}, {}, {}),
        # Learned positions, untied linear heads, a shared residual, plain learned upsampling,
        # a two-block decoder and sequence compression.
        "b": (dict(use_rope=False, tie_heads_to_codebook=False, use_level_encoding=False, use_qk_norm=False),
              dict(dec_layers=2, use_qk_norm=False, compression_factor=2, context_length=32),
              dict(quant_resi=dict(share_mode=1, ratio=0.3), learned_l1_cascade=False)),
        # Trainable tied heads with a bias and no learned scale, partially shared learnable
        # residuals, interpolated level-0 upsampling, decoder positions, a narrower MLP.
        "c": (dict(tie_heads_trainable=True, tie_heads_bias=True, tie_heads_learn_scale=False, mlp_ratio=2.0),
              dict(dec_positional=True),
              dict(quant_resi=dict(share_mode=2, learnable_ratio=True), learned_l1_sampling=False)),
    }
    for name, (m_upd, vq_upd, q_upd) in variants.items():
        torch.manual_seed(0)
        model = json.loads(json.dumps(base_model))
        vq = json.loads(json.dumps(base_vq))
        model.update(m_upd)
        vq.update(vq_upd)
        qc = vq["vector_quantizer_config"]
        for k, v in q_upd.items():
            if isinstance(v, dict):
                qc[k].update(v)
            else:
                qc[k] = v
        vq_cfg = OmegaConf.create(vq)
        vqvae = VQVAE(**OmegaConf.to_container(vq_cfg, resolve=True) | {"vector_quantizer_config": vq_cfg.vector_quantizer_config})
        noisy(vqvae, 0.2)
        net = NextConceptTransformer(vqvae_local=vqvae, config=OmegaConf.create(model))
        noisy(net, 0.2)

        d = f"{out}/ncm/{name}"
        os.makedirs(d, exist_ok=True)
        save_file(tensors(net.state_dict()) | {"level0_unigram": torch.softmax(torch.randn(16), 0)},
                  f"{d}/generator.safetensors")
        save_file(tensors(vqvae.state_dict(), ("ema_", "codebook_hit_", "_km_done", "_fd_step")),
                  f"{d}/tokenizer.safetensors")
        dump_json(f"{d}/config.json", {"model": model, "vqvae": vq, "eos_token_id": 0, "pad_token_id": 1})

        l0 = torch.tensor([[3], [11], [7]])
        prefix = torch.tensor([[0, 1], [0, 5], [7, 0]])
        with torch.no_grad():
            codes = net.generate(l0, prefix, temperature=0)
            _, zl = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            ref = {"l0": l0, "prefix": prefix, "codes": codes, "logits": vqvae.decoder(zl[-1])}
            for i, z in enumerate(zl):
                ref[f"level_logits.{i}"] = vqvae.decoder(z)
        save_ref(f"{d}/reference.safetensors", ref)
    sys.path.remove(ncm_src)


# ---------------------------------------------------------------------------------------------
# Qwen3.5 and Qwen4-Exp
# ---------------------------------------------------------------------------------------------

def run_lm(model, vocab, batch, prompt_len, steps, cont_len, ids=None):
    """Prefill, greedy cached decode steps, then a multi-token continuation."""
    ids = torch.randint(0, vocab, (batch, prompt_len)) if ids is None else ids
    ref = {"prompt": ids}
    with torch.no_grad():
        o = model(ids, use_cache=True)
        ref["prefill_logits"] = o.logits
        cache = o.past_key_values
        toks = o.logits[:, -1].argmax(-1, keepdim=True)
        steps_toks = []
        for i in range(steps):
            steps_toks.append(toks)
            o = model(toks, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            ref[f"decode_logits.{i}"] = o.logits
            toks = o.logits[:, -1].argmax(-1, keepdim=True)
        ref["decode_tokens"] = torch.cat(steps_toks, 1)
        cont = torch.randint(0, vocab, (batch, cont_len))
        ref["cont"] = cont
        ref["cont_logits"] = model(cont, past_key_values=cache, use_cache=True).logits
    return ref


def qwen3_5_fixtures(out):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    for name, multimodal in (("text", False), ("multimodal", True)):
        torch.manual_seed(0)
        cfg = Qwen3_5TextConfig(
            vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=multimodal,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        )
        model = noisy(Qwen3_5ForCausalLM(cfg), 0.05)
        sd = tensors(model.state_dict())
        config = cfg.to_dict()
        if multimodal:
            # The multimodal layout: weights under model.language_model, a nested text config with
            # tied embeddings, layer types left to full_attention_interval, and the rope settings
            # at the top level as in older configs.
            sd = {k.replace("model.", "model.language_model.", 1): v for k, v in sd.items() if k != "lm_head.weight"}
            del config["layer_types"]
            rope = config.pop("rope_parameters")
            config["rope_theta"] = rope["rope_theta"]
            config["partial_rotary_factor"] = rope["partial_rotary_factor"]
            config = {"model_type": "qwen3_5", "text_config": config, "tie_word_embeddings": True}
        d = f"{out}/qwen3_5/{name}"
        os.makedirs(d, exist_ok=True)
        save_file(sd, f"{d}/model.safetensors")
        dump_json(f"{d}/config.json", config)
        save_ref(f"{d}/reference.safetensors", run_lm(model, 50, 2, 70, 2, 20))


def qwen4_exp_fixtures(out):
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    torch.manual_seed(0)
    cfg = Qwen4ExpTextConfig(
        vocab_size=50, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        moe_intermediate_size=16, shared_expert_intermediate_size=24, num_experts=4, num_experts_per_tok=2,
        hc_count=2, hc_lowrank=8,
        ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=3, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=200, make_ngram_vocab_size_divisible_by=16, split_ngram_parts=2,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
        output_gate_type="sigmoid", eos_token_id=[3, 5], tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    )
    model = noisy(Qwen4ExpForCausalLM(cfg), 0.05)
    d = f"{out}/qwen4_exp"
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    for f in ("generation_config.json",):
        if os.path.exists(f"{d}/{f}"):
            os.remove(f"{d}/{f}")
    ids = torch.randint(0, 50, (2, 60))
    ids[0, 20] = 3  # end-of-text tokens reset the n-gram window
    ids[1, 45] = 3
    save_ref(f"{d}/reference.safetensors", run_lm(model, 50, 2, 60, 2, 20, ids))


# ---------------------------------------------------------------------------------------------
# FLUX.2
# ---------------------------------------------------------------------------------------------

def flux2_fixtures(out):
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import MistralConfig, MistralModel, Qwen3Config, Qwen3ForCausalLM

    d = f"{out}/flux2"
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(0)
    ref = {}

    def save_module(module, name, config):
        save_file(tensors(module.state_dict()), f"{d}/{name}.safetensors")
        dump_json(f"{d}/{name}.json", config)

    # Text encoders: hidden states 1, 2 and 3 (the final-normed output) of a padded batch.
    ids = torch.randint(0, 40, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    mask[1, 8:] = 0
    ids[1, 8:] = 0
    ref["te_ids"], ref["te_mask"] = ids, mask
    common = dict(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=3, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, max_position_embeddings=64, attn_implementation="eager")
    qwen = noisy(Qwen3ForCausalLM(Qwen3Config(**common, tie_word_embeddings=False)), 0.2)
    save_module(qwen, "qwen3", qwen.config.to_dict())
    mistral = noisy(MistralModel(MistralConfig(**common, sliding_window=None, rope_theta=1e6)), 0.2)
    save_module(mistral, "mistral", mistral.config.to_dict())
    with torch.no_grad():
        hq = qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        hm = mistral(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    for k in (1, 2, 3):
        ref[f"qwen3_hidden.{k}"] = hq[k]
        ref[f"mistral_hidden.{k}"] = hm[k]

    # Transformer, with and without the guidance embedding.
    tcfg = dict(in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=48, timestep_guidance_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
                rope_theta=2000)
    img = torch.randn(2, 16, 16)
    txt = torch.randn(2, 12, 48)
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(4), torch.arange(4), torch.arange(1))
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(12))
    t = torch.tensor([0.7, 0.25])
    ref.update(img=img, txt=txt, timestep=t)
    for name, guided in (("transformer_guided", True), ("transformer", False)):
        m = noisy(Flux2Transformer2DModel(**tcfg, guidance_embeds=guided), 0.05)
        save_module(m, name, dict(m.config))
        with torch.no_grad():
            ref[f"{name}_out"] = m(hidden_states=img, encoder_hidden_states=txt, timestep=t, img_ids=img_ids,
                                   txt_ids=txt_ids, guidance=torch.tensor([3.5, 2.0]) if guided else None,
                                   return_dict=False)[0]
        if not guided:
            transformer = m

    # VAE with batch-norm statistics.
    vae = AutoencoderKLFlux2(block_out_channels=(8, 16), layers_per_block=1, latent_channels=4, norm_num_groups=4,
                             down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2)
    noisy(vae, 0.02)
    with torch.no_grad():
        vae.bn.running_mean.copy_(0.3 * torch.randn(16))
        vae.bn.running_var.copy_(torch.rand(16) + 0.5)
    save_module(vae, "vae", dict(vae.config))
    z = torch.randn(2, 16, 4, 4)
    pix = torch.rand(2, 3, 16, 16) * 2 - 1
    with torch.no_grad():
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        ref["vae_decoded"] = vae.decode(Flux2KleinPipeline._unpatchify_latents(z * std + mean), return_dict=False)[0]
        enc = Flux2KleinPipeline._patchify_latents(vae.encode(pix).latent_dist.mode())
        ref["vae_encoded"] = (enc - mean) / std
    ref.update(vae_z=z, vae_pixels=pix)

    # The klein pipeline from prompt embeddings: step-distilled, then with classifier-free guidance.
    sched_cfg = dict(use_dynamic_shifting=True, time_shift_type="exponential")
    dump_json(f"{d}/scheduler.json", sched_cfg)
    latents = torch.randn(1, 16, 4, 4)
    neg = torch.randn(1, 12, 48)
    ref.update(pipe_latents=latents, pipe_prompt=txt[:1].clone(), pipe_negative=neg)
    for name, distilled, g in (("pipe_distilled", True, 1.0), ("pipe_cfg", False, 3.0)):
        pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler(**sched_cfg), vae=vae, text_encoder=qwen,
                                  tokenizer=None, transformer=transformer, is_distilled=distilled)
        kw = {} if distilled else dict(negative_prompt_embeds=neg)
        with torch.no_grad():
            ref[f"{name}_image"] = pipe(prompt_embeds=txt[:1], latents=latents, height=16, width=16,
                                        num_inference_steps=3, guidance_scale=g, output_type="pt", **kw).images
        ref[f"{name}_sigmas"] = pipe.scheduler.sigmas
    save_ref(f"{d}/reference.safetensors", ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm-src", help="a checkout of HazyResearch/ncm; the NCM fixtures are skipped without it")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    if args.ncm_src:
        ncm_fixtures(args.out, args.ncm_src)
    qwen3_5_fixtures(args.out)
    qwen4_exp_fixtures(args.out)
    flux2_fixtures(args.out)


if __name__ == "__main__":
    main()
