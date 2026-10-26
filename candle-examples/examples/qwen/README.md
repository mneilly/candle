# candle-qwen: large language model series from Alibaba Cloud

Qwen 1.5 is a series of large language models that provide strong performances
on English and Chinese.

- [Blog post](https://qwenlm.github.io/blog/qwen1.5/) introducing Qwen1.5.
- [Model card](https://huggingface.co/Qwen/Qwen1.5-0.5B) on the HuggingFace Hub.
- [Blog post](https://qwenlm.github.io/blog/qwen-moe/) for the
  mixture-of-experts (MoE) variant.

## Running the example

```bash
$ cargo run --example qwen --release  -- --prompt "Hello there "
```

Various model sizes are available via the `--model` argument, including the MoE
variant.

```bash
$ cargo run --example qwen --release  -- --model moe-a2.7b --prompt 'def print_prime(n: int): '
def print_prime(n: int):  # n is the number of primes to be printed
    for i in range(2, n + 1):
        if all(i % j != 0 for j in range(2, i)):
            print(i)
```

The qwen3 MoE variant is also an option.

```bash
$ cargo run --example qwen --features metal --release  -- --prompt "Write a poem about butterflies. <think></think>." --model "3-moe-a3b"
> In morning's hush, where daisies sleep,  
> A fleeting dance through sunlit deep—  
> They flutter soft on gossamer thread,  
> The messengers of spring’s own head.
> 
> With painted sails and delicate grace,  
> They drift from bloom to blossom's face.  
> Each wing a tale in hues unseen,  
> Of ancient dreams and secrets between.
> 
> No sound they make, yet still they speak—  
> Of time that flies, of life so brief.  
> A fleeting kiss on summer’s breath,  
> A whisper lost before death.
> 
> Yet in their flight, the soul takes wing,  
> And for a moment, all is spring.  
> For though they fade, they never die—  
> Their beauty lives where hearts can fly.
> 161 tokens generated (3.00 token/s)
```

```shell
# Local unquantized 32B MoE model (with Fused MoE kernel) (~80GB GPU memory)
cargo run --example qwen --features cuda --release  -- --prompt "Write a poem about butterflies. <think></think>." --model "3-moe-a3b" --weight-path /path/Qwen3-30B-A3B-Instruct-2507
```
## Qwen3.5 and Qwen3.8

The dense Qwen3.5 and Qwen3.8 checkpoints use a hybrid architecture: three layers out of four
use Gated DeltaNet, a linear-attention recurrence with a fixed-size state, and the others use
gated full attention. Only the text model is supported; the vision tower and the
multi-token-prediction weights of these checkpoints are not loaded.

```bash
$ cargo run --example qwen --features cuda --release -- --model 3.8-27b --prompt "Explain the delta rule."
```

The 27B models need about 54GB of GPU memory in bf16. As for Qwen3, the prompt is wrapped in the
chat template unless `--no-chat-template` is passed, and `--thinking` lets the model reason before
answering.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next previews the Qwen4 architecture: a 125B-parameter mixture of experts with 6B
active parameters per token. On top of the Gated DeltaNet hybrid, it widens the residual stream
into four gated streams, makes the full-attention layers sparse with Qwen Sparse Attention, and
adds a 51B-parameter hashed n-gram embedding on one layer. Only the text model is supported.

```bash
$ cargo run --example qwen --features cuda --release -- --model 3.8-flash-next --prompt "Explain the delta rule."
```

The n-gram embedding table stays memory-mapped on the host, and only the rows each step needs
are read from it. The rest of the model, about 250GB in bf16, is loaded on the device.
