# Candle: software architecture

This document describes how candle is organized: its components, their responsibilities, how they
depend on each other, and the design decisions that shape them. How each part is built in code is
described separately in [implementation.md](implementation.md).

## Purpose and design principles

Candle is a machine-learning framework for Rust, aimed first at inference, with training
supported. Its architecture follows a few principles:

- **Small core, everything else built on it.** One crate defines tensors and their operations;
  neural-network layers, models and applications are ordinary libraries on top.
- **Eager execution.** Operations run when they are called. There is no graph compiler, no
  ahead-of-time tracing and no separate "session": a model is plain Rust code calling tensor
  operations.
- **One API, several devices.** The same tensor code runs on the CPU, on NVIDIA GPUs (CUDA) and on
  Apple GPUs (Metal). The device is a value carried by each tensor, not a type parameter.
- **Optional heavy dependencies.** GPU support, vendor math libraries and kernel libraries are
  opt-in at build time, so the default build is pure Rust and small enough for WebAssembly.
- **Interoperate with the Python ecosystem's formats.** Weights are read from the formats that
  PyTorch, Hugging Face and llama.cpp produce, rather than through a candle-specific format.

## Layers

```
 Applications     candle-examples   candle-wasm-examples   candle-pyo3   tensor-tools
                         |                   |                  |             |
 Model zoo        candle-transformers  (models, generation, pipelines, quantized layers)
                         |
 NN building      candle-nn            (layers, parameter loading, ops, optimizers)
 blocks                  |
 Tensor core      candle-core          (tensors, devices, autograd, quantization, file formats)
                    /    |     \
 Kernels   candle-kernels  candle-metal-kernels  candle-flash-attn(-v3)   (+ gemm, cuBLAS, MKL…)

 Side crates      candle-onnx (ONNX graphs on candle-core)   candle-datasets (data loading)
```

Each layer depends only on the layers below it. The kernel crates are leaves: they know nothing
about tensors and only expose compiled GPU code and launch helpers.

### candle-core: the tensor core

Owns the concepts every other crate builds on:

- **Tensor**: an immutable, cheaply cloned handle to an n-dimensional array. Operations return new
  tensors; views (reshape, transpose, narrow, broadcast) share the underlying data.
- **Device**: where the data lives and where operations run (CPU, a CUDA device, a Metal device).
- **Storage**: the device-specific buffer behind a tensor.
- **Layout and Shape**: how a tensor's elements map onto its storage (shape, strides, offset).
- **DType**: the element type (integers, f16/bf16/f32/f64, and 8/6/4-bit floats).
- **Operations**: the fixed vocabulary of primitive operations every backend must provide.
- **Autograd**: reverse-mode differentiation over the operations that produced a tensor.
- **Quantized tensors**: a parallel tensor type for block-quantized weights (the GGML/GGUF
  formats), with its own storage and matrix multiplication.
- **Serialization**: readers and writers for safetensors, NumPy, PyTorch pickles and GGUF/GGML.

### candle-nn: neural-network building blocks

Turns tensor operations into reusable layers and gives models a uniform way to obtain parameters:

- **Module**: the single-method `forward` interface that layers and models implement.
- **Layers**: linear, embedding, convolutions, normalizations, RNNs, rotary embeddings, attention
  helpers, mixture-of-experts helpers, KV caches.
- **Fused ops**: operations such as softmax, RMS norm, layer norm and scaled-dot-product
  attention, with dedicated per-device kernels where available.
- **VarBuilder**: the parameter-loading abstraction. A model asks for a tensor by hierarchical
  name and shape; where it comes from (a safetensors file, a memory map, an in-memory map, fresh
  initialization for training, zeros) is decided by the builder's backend.
- **Training support**: variable maps, initializers, losses, optimizers (SGD, AdamW).

### candle-transformers: the model zoo

A library of model implementations (language, vision, audio, diffusion and multimodal models),
each written against candle-nn and candle-core. Alongside the models it holds what they share:
text generation (sampling and logits processing), pipelines, quantized variants of common layers,
a quantized parameter loader for GGUF files, tracing wrappers and attention/masking utilities.

Models follow a common shape: a configuration type parsed from the checkpoint's JSON, a
constructor taking the configuration and a VarBuilder, and forward methods that take token ids and
a position offset. Models that generate text keep their own KV caches and recurrent states
internally.

### Applications and bindings

- **candle-examples**: command-line programs, one per model or task, plus shared helpers
  (downloading from the Hugging Face Hub, tokenizer streaming, audio and image utilities, device
  selection).
- **candle-wasm-examples**: models compiled to WebAssembly for the browser.
- **candle-pyo3**: Python bindings exposing tensors and a subset of the API.
- **candle-onnx**: evaluates ONNX graphs by mapping ONNX operators onto candle operations.
- **candle-datasets**: dataset loading and batching for training examples.
- **tensor-tools**: a command-line tool to inspect and convert weight files.

### Kernel crates

- **candle-kernels**: CUDA kernels for the core operations and quantized matmuls, compiled at
  build time.
- **candle-metal-kernels**: Metal shader sources and the Rust code that compiles and dispatches
  them.
- **candle-flash-attn, candle-flash-attn-v3**: FlashAttention for CUDA, as separate optional
  crates.

## Core architectural decisions

### Device dispatch through enums, not generics

A tensor holds its device and storage as enums with one variant per backend. Every operation is
written once at the tensor level, validates its arguments (shapes, dtypes, same device) and
dispatches to the backend that owns the data. Backends implement one common backend interface. As
a result:

- model code is device-agnostic and needs no type parameters;
- mixing devices in one operation is a runtime error, not a compile error;
- adding a backend means implementing that interface and adding a variant, without touching
  models.

### Views and copy-on-demand

A tensor separates *what* its data is (shared storage) from *how* it is seen (its layout). Shape
operations create new layouts over the same storage, so they cost nothing. Operations that need
contiguous data (most matmuls, some kernels) ask for it explicitly, and a copy is made only then.

### Eager execution with asynchronous devices

Calls run in program order. GPU backends enqueue work on device queues and return immediately;
synchronization happens only when data is read back to the host or explicitly requested. The CPU
backend runs synchronously and parallelizes inside operations.

### Autograd as an optional layer on eager execution

Gradients are not a separate mode. A tensor remembers the operation that produced it only when one
of its inputs requires gradients (a *variable*); otherwise nothing is recorded, so inference pays
no autograd cost. Calling `backward` on a result walks this recorded graph and returns a gradient
store keyed by tensor identity, which optimizers consume.

### Two tensor families: dense and quantized

Dense tensors cover the general operation set. Quantized tensors store weights in block formats
(GGML's Q4_0 … Q8_K, and others) and support only what inference needs from weights: matrix
multiplication against dense activations, row lookup (embeddings) and dequantization. A
`QMatMul` layer hides the choice, so a model can mix quantized and dense weights.

### Extension points

- **Custom operations**: users add operations with per-backend implementations (and optionally a
  backward pass) without modifying candle-core. candle-nn's fused ops use this mechanism.
- **Weight sources**: a new VarBuilder backend adds a parameter source.
- **New backends**: implement the backend interface and add a device variant.
- **Models**: plain libraries against the public API; nothing registers them centrally.

### Build-time composition

Cargo features select the capabilities compiled in: `cuda` (with optional `cudnn`, `nccl`,
`cutile`), `metal`, `mkl`/`accelerate` (vendor math libraries for the CPU), flash attention, and
example-specific features. When a backend is not compiled in, a stub with the same interface takes
its place, so the rest of the code compiles unchanged and the backend reports itself as
unavailable at runtime.

## Data flow

### Loading a model

```
checkpoint files ──► format reader (safetensors / GGUF / pickle) ──► VarBuilder backend
      config.json ──► model Config                                          │
                                   Model::new(&config, var_builder) ◄────────┘
                                   (asks for each parameter by name and shape)
```

Parameters are read lazily, on request, so a model reads only the tensors it asks for. A memory
mapping means that weights never requested are never read.

### Running a forward pass

```
token ids ─► embedding ─► layers (attention / MLP / ...) ─► norm ─► LM head ─► logits
                               │            ▲
                               └ KV caches, recurrent state (inside the model)
logits ─► LogitsProcessor (temperature, top-k/p, penalties) ─► next token ─► next forward call
```

Each step calls tensor operations that dispatch to the device holding the weights.

### Training

```
Var parameters ─► forward ─► loss ─► backward() ─► GradStore ─► Optimizer::step ─► updated Vars
```

## Cross-cutting concerns

- **Errors**: one error type for the whole stack, with context (shape mismatches, device and dtype
  mismatches, missing tensors) and optional backtraces. Fallible operations return `Result`.
- **Concurrency**: tensors can be shared across threads. Storage is guarded for the rare in-place
  mutations (variables, KV caches), and CPU operations use internal thread pools.
- **Precision**: model code chooses the dtype. A common pattern is reduced precision (bf16/f16) on
  GPUs, f32 on CPUs, and f32 for numerically sensitive steps (softmax, norms, recurrences).
- **Observability**: optional `tracing` spans around layers, and Chrome trace output in examples.
- **Portability**: the default build is pure Rust, and runs on Linux, macOS, Windows and
  WebAssembly.
