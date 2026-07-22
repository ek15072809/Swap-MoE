<div align="center">

# Swap-MoE

### On-demand SSD loading for massive MoE models on low-RAM machines

**Run 200B+ parameter Mixture-of-Experts models on a 16GB RAM consumer PC — no GPU required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-1f66c3c-blue)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)]()

[Overview](#overview) •
[How it works](#how-it-works) •
[Installation](#installation) •
[Usage](#usage) •
[Benchmarks](#benchmarks) •
[Limitations](#known-limitations) •
[Report](#technical-report) •
[License](#license) •
[Technical Report (PDF)](docs/Swap-MoE_Technical_Report.pdf)

</div>

---

## Overview

Modern Mixture-of-Experts (MoE) language models pack hundreds of billions of
total parameters, but only route a small fraction of them — the *active
parameters* — to compute any single token. Despite this, virtually every
existing inference engine still assumes the **entire** model file can be
resident in physical RAM. In practice, this means a MoE model whose file size
exceeds your machine's RAM simply refuses to load, even though most of that
file is never touched for any given token.

**Swap-MoE** is a small patch set for [llama.cpp](https://github.com/ggml-org/llama.cpp)
that removes this restriction. It keeps the weights of routed (non-shared)
experts `mmap`'d directly on the SSD and lets the OS's own demand-paging
mechanism pull only the experts that are actually selected by the router into
RAM, on the fly, during inference. Two optional companion mechanisms —
**LRU keep-recent** and **Expert Fixation** — give you further control over
RAM usage and SSD I/O.

With Swap-MoE, all three MoE models below — none of which could even *start*
under normal `llama.cpp` loading on a 16GB-RAM machine — become runnable at
practical token rates:

| Model | Quant | File size | Normal load | Swap-MoE |
|---|---|---|---|---|
| Qwen3.5-35B-A3B    | Q4_K_M | 19.7 GB | ❌ fails to launch | ✅ 7.28 tok/s |
| Qwen3.5-122B-A10B  | Q2_K   | 42.6 GB | ❌ fails to launch | ✅ 2.00 tok/s |
| MiniMax-2.7        | IQ1_M  | 60.7 GB | ❌ fails to launch | ✅ 0.75 tok/s |

See [Benchmarks](#benchmarks) below and the [technical report](#technical-report)
for the full methodology and additional configurations.

## How it works

### 1. Expert Streaming — on-demand paging via `mmap`

llama.cpp's normal loader touches every byte of every tensor at least once
during startup (for validation and backend upload), which defeats the lazy
loading that `mmap` would otherwise provide. Swap-MoE changes this for one
specific class of tensor: **routed experts**.

At load time, any tensor whose name matches the standard MoE naming scheme —

```
blk.<layer>.ffn_gate_exps.weight
blk.<layer>.ffn_up_exps.weight
blk.<layer>.ffn_down_exps.weight
blk.<layer>.ffn_gate_up_exps.weight   (fused variant)
```

— **and does not** carry the `_shexp` (shared expert) suffix, is classified
as a routed expert and is **not** read into RAM at load time — only its
buffer pointer is assigned. The router (`ffn_gate_inp`) and shared experts
(`ffn_*_shexp`) are excluded from this and always loaded as normal, since
they're needed on every single token regardless of routing.

The corresponding `mmap` pages are then explicitly marked non-resident
(`madvise` on Linux, the equivalent page API on Windows), even if the OS
happened to page some of them in during the mapping step. From that point on,
whenever the router actually selects an expert during a forward pass, the
resulting page fault pulls just that expert's weights off the SSD — nothing
more.

This is enabled with a single flag:

```
--expert-streaming
```

Supported on any MoE architecture that uses the standard `ffn_*_exps` tensor
naming, including **Qwen3-MoE**, **Qwen3.5-MoE**, **MiniMax-M2 / M2.7**,
**DeepSeek-V2 / V3**, **Laguna-S-2.1** and so on.

### 2. LRU keep-recent — explicit RAM budget per layer

Left alone, the OS page cache will happily let paged-in expert weights
accumulate in RAM for as long as memory pressure allows, which can starve
other processes or trigger reclaim storms during long generations. `--expert-keep-recent N`
adds a per-layer LRU list that tracks the *N* most recently used experts and
explicitly evicts (`MADV_DONTNEED` / `VirtualUnlock`) any expert that falls
outside that window after every forward pass.

```
--expert-keep-recent 12
```

This is only meaningful together with `--expert-streaming`. Note: in our
measurements, explicit eviction consistently ran *slower* than simply letting
the OS manage its own page cache — see [Benchmarks](#benchmarks). Use it only
when you need a hard cap on resident RAM, not for speed.

### 3. Expert Fixation — reducing SSD I/O by freezing routing

Even with Expert Streaming, MoE routing can change every token, meaning SSD
reads (page faults) never stop during generation. **Expert Fixation** trades
some routing accuracy for a large reduction in SSD I/O: for a window of *N*
forward passes, it pins each layer's router output to the expert IDs chosen
on the first (non-warmup) pass in that window, forcing the *same* experts to
be reused. Since the same pages are hit repeatedly, everything after the
first access lands in the OS page cache and no further SSD reads occur until
the window expires.

```
--expert-fixation 4
```

Only meaningful together with `--expert-streaming`. It is implemented as an
ordinary `GGML_OP_MAP_CUSTOM1` node inserted right after the router's top-k
selection — **not** as a scheduler callback — so the compute graph's shape
never changes whether fixation is active, expired, or disabled, and it stays
fully compatible with llama.cpp's graph-reuse optimization. See the
[technical report](#technical-report) for full implementation details
(thread-safety, warmup handling, interaction with `--expert-keep-recent`).

`--expert-fixation` is fully orthogonal to `--override-kv`: the latter
changes model metadata at load time (e.g. how many experts are routed per
token), while fixation overrides the *runtime* top-k choice. They are often
combined to offset fixation's accuracy cost by routing to more experts:

```
--expert-streaming --expert-fixation 4 \
  --override-kv "minimax-m2.expert_count=int:256,minimax-m2.expert_used_count=int:12"
```

**Fixation can degrade generation quality**, sometimes severely, depending on
model, window length, and quantization — see [Benchmarks](#benchmarks) and
[Known Limitations](#known-limitations).

## Installation

Swap-MoE is distributed as a single patch file, `llama.cpp-expert-streaming.patch`,
against a specific llama.cpp commit. It is **not** a standalone fork — you
apply it to your own llama.cpp checkout.

> **This release targets llama.cpp commit `1f66c3c` ("Add support for Laguna XS.2 & M.1", July 2026).
> The previous release targeted `6cb18b2f`.**

### Requirements

- A C++17 toolchain (GCC/Clang on Linux, MSVC or clang-cl on Windows) capable
  of building llama.cpp
- CMake ≥ 3.14
- `git`
- An NVMe SSD is strongly recommended (SATA SSDs will work but with lower
  throughput; spinning disks are not recommended)
- Enough RAM to comfortably hold the *non-expert* part of the model plus your
  desired working set of experts — **not** the full model file

### Steps

```bash
# 1. Clone llama.cpp and check out the exact commit this patch targets
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 1f66c3c

# 2. Apply the Swap-MoE patch
git apply /path/to/llama.cpp-expert-streaming.patch

# (if git apply complains about whitespace/context drift, try:)
# git apply --3way /path/to/llama.cpp-expert-streaming.patch
# patch -p1 < /path/to/llama.cpp-expert-streaming.patch

# 3. Configure and build (CPU-only build; see llama.cpp's own docs for
#    GPU-backend flags — GPU offload is not currently combined with
#    Expert Fixation, see Known Limitations)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

# The CLI binary will be at build/bin/llama-cli (Linux/macOS)
# or build\bin\Release\llama-cli.exe (Windows)
```

> **Note:** This patch must be applied to llama.cpp at commit `1f66c3c`.
> Applying it to a different commit may fail or silently produce incorrect
> results, since it touches internal loader and graph-building code that
> changes frequently upstream.
>
> **What changed vs the previous release (`6cb18b2f`):**
> - `llm_graph_context::build_moe_ffn` now accepts an optional
>   `selected_experts_in` parameter that, when non-null, bypasses the
>   `ggml_argsort_top_k` call. The fixation graph node is therefore inserted
>   *after* the (possibly pre-supplied) selected-experts tensor rather than
>   unconditionally after `argsort_top_k`.
> - `llama_model::impl` gained a `tensor_split_owned` member upstream; the
>   expert-streaming and expert-fixation state is now placed *after* that new
>   member so the struct layout stays consistent.
> - All other hunks apply cleanly with a small line-offset shift.
>
> These are pure porting fixes — no behavioural change vs the `6cb18b2f`
> release. Verified by building llama.cpp `1f66c3c` with the patch applied
> (CPU-only Release build) and exercising the patched code paths with a
> synthetic Qwen2-MoE GGUF:
> - `--expert-streaming` triggers `expert streaming enabled — skipping initial
>   mmap prefetch for routed expert tensors` and `expert_stream: tracking 24
>   routed expert slices`.
> - `--expert-keep-recent 2` triggers `expert keep-recent window = 2 experts
>   per layer`.
> - `--expert-fixation 4` triggers `expert fixation active — top-k pinned for
>   the first 4 forward passes`, and the per-layer fixation graph nodes are
>   inserted (graph node count rises from 106 → 108 for a 2-layer test model).
> - `--expert-cache-size` is accepted by both `llama-cli` and `llama-server`.

## Usage

All flags below are added on top of llama.cpp's normal CLI (`llama-cli`,
`llama-server`, etc. — anything that goes through `common_params`).

| Flag | Argument | Default | Description |
|---|---|---|---|
| `--expert-streaming` | — | off | Skip prefetching routed expert weights at startup; the OS demand-pages them from SSD during MoE forward. Implies `--mmap` and is incompatible with `--mlock` (both are forced automatically, with a warning, if needed). |
| `--expert-cache-size` | `N_MiB` | `0` | Target RAM (MiB) for the routed-expert page cache. `0` lets the OS manage it. When set, a byte-budget eviction pass runs after every token. |
| `--expert-keep-recent` | `N` | `0` | Keep the *N* most recently used experts per layer resident, explicitly evicting anything older after each forward pass. `0` disables this. Only meaningful with `--expert-streaming`. |
| `--expert-fixation` | `N` | `0` | Pin the router's top-k selection to the first pass's result for the next *N* forward passes, eliminating repeated SSD reads for those tokens. `0` disables this. Only meaningful with `--expert-streaming`. |

Each flag also has a corresponding environment variable
(`LLAMA_ARG_EXPERT_STREAMING`, `LLAMA_ARG_EXPERT_CACHE_SIZE`,
`LLAMA_ARG_EXPERT_KEEP_RECENT`, `LLAMA_ARG_EXPERT_FIXATION`).

### Example: minimal — just make a huge model launch

```bash
./build/bin/llama-cli \
  -m /path/to/MiniMax-2.7-IQ1_M.gguf \
  --expert-streaming \
  -p "Hello, world!" \
  -n 128
```

### Example: cap resident expert RAM explicitly

```bash
./build/bin/llama-cli \
  -m /path/to/Qwen3.5-122B-A10B-Q2_K.gguf \
  --expert-streaming \
  --expert-keep-recent 12 \
  -p "Explain the theory of relativity." \
  -n 256
```

### Example: reduce SSD I/O with fixation, compensated by wider routing

```bash
./build/bin/llama-cli \
  -m /path/to/minimax-m2.gguf \
  --expert-streaming \
  --expert-fixation 4 \
  --override-kv "minimax-m2.expert_count=int:256,minimax-m2.expert_used_count=int:12" \
  -p "Write a short story about a lighthouse keeper." \
  -n 512
```

### Example: `llama-server`

The same flags work with `llama-server`:

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --expert-streaming \
  --expert-fixation 2 \
  --host 0.0.0.0 --port 8080
```

### Choosing values

- Start with `--expert-streaming` alone. It has no accuracy impact and is
  what unlocks running models that would otherwise fail to launch.
- Only add `--expert-keep-recent` if you need a hard RAM ceiling; expect a
  moderate speed penalty compared to letting the OS manage its own cache.
- Only add `--expert-fixation` if SSD I/O — not RAM — is your bottleneck
  (e.g. a slower SATA SSD), and validate output quality for your specific
  model and window size before relying on it. Small active-parameter, higher-bit
  models tend to tolerate small fixation windows (e.g. `N=2`) reasonably well;
  large active-parameter or very low-bit quantized models degrade quickly.

## Benchmarks

Measured on a consumer PC: 13th-gen Intel Core i7, 16GB physical RAM, NVMe
SSD, CPU-only inference (no GPU offload). Throughput is instantaneous
tokens/s measured at decode token 400.

| Model | Quant | Size (GB) | normal | expert-streaming | keep-recent 12 | fixation N=2 | fixation N=4 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B   | Q4_K_M | 19.7 | ❌ fails to launch | 7.01 | 5.89 | **7.28** | 5.99 ⚠️ |
| Qwen3.5-122B-A10B | Q2_K   | 42.6 | ❌ fails to launch | 2.00 | 1.00 | 0.35 ⚠️ | 0.53 ⚠️ |
| MiniMax-2.7       | IQ1_M  | 60.7 | ❌ fails to launch | 0.75 | 0.44 | 0.53 ⚠️ | 0.40 ⚠️ |

⚠️ = a numeric throughput value was measured, but generation quality was
visually confirmed to have collapsed (repetitive/incoherent output) under
this condition — the speed number is **not** representative of usable
output. See the [technical report](#technical-report) for the full
discussion.

**Key takeaways:**

- Expert Streaming alone turns "does not launch" into 0.75–7.01 tok/s across
  all three models, with no observed quality impact.
- `--expert-keep-recent` was slower than plain `expert-streaming` in every
  test here — explicit eviction costs more than it saves compared to the
  OS's own page-cache management, unless you specifically need a RAM cap.
- `--expert-fixation` only paid off cleanly on the smallest/highest-active-parameter
  model (Qwen3.5-35B-A3B at N=2). On larger or more aggressively quantized
  models, fixation reliably degraded output quality even though the
  raw token rate looked fine.

## Known Limitations

- **CPU backend only.** The Expert Fixation graph node is implemented as
  `GGML_OP_MAP_CUSTOM1` and currently runs only on the CPU backend. Combining
  it with GPU offload (`-ngl`) is not supported.
- **Single-context usage assumed.** Fixation state (counter + cache) is tied
  to the model object. If multiple inference contexts share one model
  instance, that state is shared across them.
- **Fixation can hurt generation quality**, sometimes severely, depending on
  model size, active-parameter count, quantization bit-width, and window
  length *N*. Reducing *N*, or increasing the number of active experts via
  `--override-kv`, may help — but no systematic tuning has been done beyond
  the configurations in the benchmark table above.
- **Single-machine evaluation.** All numbers above come from one hardware
  configuration. Results will vary with different CPUs and, especially,
  different SSD read latency/throughput characteristics.
- This patch is **experimental** and modifies internal loader and graph-build
  code paths in llama.cpp. Expect to need to re-port it when updating to a
  newer llama.cpp commit.

## Technical Report

A full write-up — motivation, design, implementation details (including
thread-safety and warmup handling for the fixation graph node), and the
complete experimental methodology — is included in this repository as
[`Swap-MoE_report.pdf`](docs/Swap-MoE_report.pdf).

## Contributing

Issues and pull requests are welcome — in particular:

- Ports of this patch to newer llama.cpp commits
- GPU-backend support for Expert Fixation
- Additional benchmark data on different hardware (CPU architectures, SSD
  types) or additional MoE models
- Improved fixation strategies that reduce the speed/quality trade-off

## Citation

If TaQuants is useful in your work, please consider citing it:

```bibtex
@software{swapmoe2026,
  author = {ek15072809},
  title  = {Swap-MoE: On-Demand SSD Loading for Inference of Massive MoE Models in Low-RAM Environments},
  year   = {2026},
  url    = {https://github.com/ek15072809/swap-moe}
}
```

## License

Swap-MoE is released under the [MIT License](./LICENSE).

This project is a patch against, and intended for use with,
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License,
© the ggml-org contributors). No llama.cpp source code is redistributed in
this repository beyond the diff itself.

## Author

**ek15072809**
