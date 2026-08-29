<div align="center">

# Swap-MoE

### On-demand SSD loading for massive MoE models on low-RAM machines

**Run 200B+ parameter Mixture-of-Experts models on a 16GB RAM consumer PC — no GPU required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-f5e85d43a-blue)](https://github.com/ggml-org/llama.cpp)
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
RAM, on the fly, during inference. Three optional companion mechanisms —
**LRU keep-recent**, **Expert Fixation**, and **Expert Prefetch** — give you
further control over RAM usage and SSD I/O.

With Swap-MoE, all four MoE models below — none of which could even *start*
under normal `llama.cpp` loading on a 16GB-RAM machine — become runnable at
practical token rates:

| Model | Quant | File size | Normal load | Swap-MoE |
|---|---|---|---|---|
| Qwen3.5-35B-A3B    | Q4_K_M | 19.7 GB | ❌ fails to launch | ✅ 7.28 tok/s |
| Qwen3.5-122B-A10B  | Q2_K   | 42.6 GB | ❌ fails to launch | ✅ 2.00 tok/s |
| MiniMax-2.7        | IQ1_M  | 60.7 GB | ❌ fails to launch | ✅ 0.75 tok/s |
| DeepSeek-V4-Flash-0731 | IQ1_M | 80.9 GB | ❌ fails to launch | ✅ 0.55 tok/s |
| Qwen3.8-Flash-Next | IQ1_M | 69.4 GB | ❌ fails to launch | ✅ 0.63 tok/s |


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

Supported on any MoE architecture that uses the standard `ffn_*_exps.weight` tensor
naming, including **Qwen3.8-Flash-Next (qwen4exp)**, **Qwen3-MoE**, **Qwen3.5-MoE**,
**MiniMax-M2 / M2.7**, **DeepSeek-V2 / V3 / V4**, **Laguna-S-2.1** and so on.
Note that DeepSeek-V4's hash-routed layers have some caveats around Expert Fixation
and Expert Prefetch — see
[Expert Prefetch](#4-expert-prefetch--hiding-ssd-latency-behind-compute) below.

> **Qwen3.8-Flash-Next (`qwen4exp`) notes:** all of its layers are MoE layers built
> through llama.cpp's standard `build_moe_ffn()` router path, so all four Swap-MoE
> mechanisms (Streaming, Keep-recent, Fixation, Prefetch) work on it without any
> special-casing. Its other architectural novelties — hyper-connections (a 4x-wide
> residual stream replacing every layer norm), QSA sparse attention on the
> full-attention layers, PLE n-gram hash embeddings, and the Qwen3.5-style hybrid
> gated-delta-net + full-attention memory — are orthogonal to expert routing and
> require no Swap-MoE changes. NVFP4-style per-expert scale tensors
> (`blk.%d.ffn_*_exps.scale`) are deliberately *not* streamed: they are tiny, and
> keeping them resident avoids mis-tracking them as expert 0's weight region.

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

### 4. Expert Prefetch — hiding SSD latency behind compute

Even with Expert Streaming, the very first access to an expert on a given
forward pass is a synchronous page fault: compute for that expert stalls
until the SSD read completes. **Expert Prefetch** removes this stall by
predicting, ahead of time, which experts the *next* forward pass will need,
and warming the OS page cache for them in the background while the CPU is
still busy with the current pass.

A dedicated background thread tracks the per-layer expert IDs selected on
the most recent (non-warmup) forward pass and uses them as the prediction
for the next one — adjacent decode steps almost always route to mostly the
same experts, so this prediction is accurate in practice. Once a new set of
predicted expert IDs is available, the thread issues non-blocking OS hints
(`madvise(WILLNEED)` on Linux, `PrefetchVirtualMemory` on Windows) for those
experts' pages, letting the kernel schedule the SSD reads concurrently with
ongoing CPU compute rather than on the critical path. By the time the next
forward pass actually reaches those experts, their weights are often already
resident, making the effective load time close to zero.

This is enabled with a single flag:

```
--expert-prefetch
```

Only meaningful together with `--expert-streaming`. A few notes on its
behavior:

- The **first forward pass** after model load has no prior-pass prediction
  to work from, so it is not prefetched — ordinary demand paging handles it,
  same as without this flag.
- **DeepSeek-V4's hash-routed layers** select their experts as a function of
  the *input token*, not the hidden state, so their routing cannot be
  predicted from the previous forward pass. Expert Prefetch (and Expert
  Fixation) are automatically skipped for those specific layers; all other,
  non-hash-routed layers in the same model still benefit normally.
- Recommended especially for decode-heavy workloads on slower (e.g. SATA)
  SSDs, where the synchronous page-fault stall is the dominant cost.

```bash
./build/bin/llama-cli \
  -m /path/to/DeepSeek-V4-Flash-0731-IQ1_M.gguf \
  --expert-streaming \
  --expert-prefetch \
  -p "Hello, world!" \
  -n 128
```

## Installation

Swap-MoE is distributed as a single patch file, `llama.cpp-expert-streaming.patch`,
against a specific llama.cpp commit. It is **not** a standalone fork — you
apply it to your own llama.cpp checkout.

> **This release targets llama.cpp commit `f5e85d43a` (2026-08-28).
> The previous release targeted `221f0f635`.**

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
git checkout f5e85d43a

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

> **Note:** This patch must be applied to llama.cpp at commit `f5e85d43a`.
> Applying it to a different commit may fail or silently produce incorrect
> results, since it touches internal loader and graph-building code that
> changes frequently upstream.
>
> **What changed vs the previous release (`221f0f635`):**
> - Ported the whole patch set to llama.cpp `f5e85d43a` (2026-08-28 master).
>   Upstream churn absorbed by this port includes: the `llama_load_mode`
>   enum migration in `include/llama.h` / `common/common.*`, the
>   `build_moe_ffn()` signature growth (per-expert NVFP4 scales and biases,
>   plus the GROVEMOE grouped-routing block now sitting downstream of the
>   fixation node), the `llm_graph_params` field additions in
>   `llama-graph.h`, the `graph_params()` / `graph_compute()` reshuffles in
>   `llama-context.cpp`, and the loader's `load_all_data()` rework in
>   `llama-model-loader.cpp`. All Swap-MoE flags, semantics and log lines
>   are unchanged.
> - Added full support for **Qwen3.8-Flash-Next** (`qwen4exp`, added
>   upstream in `6c84c7d5d` — *model: add Qwen3.8-Flash-Next*). The model's
>   routed experts use the standard `ffn_*_exps.weight` naming and its
>   standard router (no hash routing), so Expert Streaming, Keep-recent,
>   Fixation and Prefetch all apply to every layer with no special-casing.
>   Its other novelties (hyper-connections, QSA sparse attention, PLE
>   n-gram hash embeddings, hybrid gated-delta-net memory) are orthogonal
>   to expert routing. See the note in
>   [Expert Streaming](#1-expert-streaming--on-demand-paging-via-mmap).
> - Hardened `is_routed_expert_tensor()`: routed-expert classification now
>   requires the `.weight` suffix (with a legacy unsuffixed fallback).
>   Previously the NVFP4 per-expert scale tensors (`ffn_*_exps.scale`)
>   and per-expert bias tensors (`ffn_*_exps.bias`, e.g. OpenAI-MoE)
>   matched the bare substring patterns. The 1-D `.scale` tensors would
>   be mis-sliced as a single "expert 0" region whose
>   `(layer, expert_id=0, tensor_kind)` key collides with the real weight
>   region in the manager's region index, silently redirecting LRU
>   eviction / prefetch of expert 0 to the wrong bytes. Those variants are
>   tiny and are now always kept resident in RAM.
> - All other hunks carried over from the `221f0f635` release.
>
> Verified by building llama.cpp `f5e85d43a` with the patch applied
> (CPU-only Release build) and exercising the patched code paths with
> synthetic GGUFs generated by llama.cpp's own `test-llama-archs` tool
> (`qwen4exp-moe`, plus `qwen2moe`, `qwen35moe` and `deepseek4` as
> regressions):
> - `--expert-streaming` triggers `expert streaming enabled — skipping initial
>   mmap prefetch for routed expert tensors` and `expert_stream: tracking 8
>   routed expert slices` (2 layers × 2 tensors × 2 experts on the test
>   model), followed by `evicting routed expert pages from RAM`.
> - `--expert-keep-recent 2` triggers `expert keep-recent window = 2 experts
>   per layer`.
> - `--expert-fixation 4` triggers `expert fixation active — top-k pinned for
>   the first 4 forward passes (graph-node based, no scheduler callback)`,
>   and per-pass logs show the countdown ticking 4 → 3 → 2 → 1 → 0.
> - `--expert-prefetch` triggers `expert prefetch enabled — background worker
>   will prefetch predicted next-token expert pages` and `async prefetch
>   thread started`.
> - A public-API decode harness (warmup pass + 12 argmax-feedback decode
>   steps) completes with rc=0 on `qwen4exp-moe` in **seven representative**
>   flag combinations spanning fixation {0, 2, 4, 8}, keep-recent {0, 2, 3}
>   and prefetch {0, 1}, including all three features combined.
> - **Numerical equivalence:** for `qwen4exp`, `qwen2moe`, `qwen35moe` and
>   `deepseek4` test models, the final logits checksum over the same
>   12-step decode is **bit-identical** with `--expert-streaming` on and
>   off — the load-time path change does not alter any computation.
> - The `LLAMA_ARG_EXPERT_STREAMING` environment variable path was
>   exercised via `llama-completion` (the CLI family shares
>   `common/arg.cpp` with `llama-cli` / `llama-server`).
> - The classification unit suite (`is_routed_expert_tensor`, 30 cases:
>   routed weights, shared experts/router, NVFP4 scales & biases, qwen4exp
>   hyper-connection/PLE tensors, DeepSeek tid2eid) passes.

## Usage

All flags below are added on top of llama.cpp's normal CLI (`llama-cli`,
`llama-server`, etc. — anything that goes through `common_params`).

| Flag | Argument | Default | Description |
|---|---|---|---|
| `--expert-streaming` | — | off | Skip prefetching routed expert weights at startup; the OS demand-pages them from SSD during MoE forward. Implies `--mmap` and is incompatible with `--mlock` (both are forced automatically, with a warning, if needed). |
| `--expert-cache-size` | `N_MiB` | `0` | Target RAM (MiB) for the routed-expert page cache. `0` lets the OS manage it. When set, a byte-budget eviction pass runs after every token. |
| `--expert-keep-recent` | `N` | `0` | Keep the *N* most recently used experts per layer resident, explicitly evicting anything older after each forward pass. `0` disables this. Only meaningful with `--expert-streaming`. |
| `--expert-fixation` | `N` | `0` | Pin the router's top-k selection to the first pass's result for the next *N* forward passes, eliminating repeated SSD reads for those tokens. `0` disables this. Only meaningful with `--expert-streaming`. |
| `--expert-prefetch` | — | off | Asynchronously prefetch the predicted next-pass routed-expert pages from SSD into the OS page cache in parallel with CPU compute, so the following forward pass finds them already resident. Only meaningful with `--expert-streaming`. Automatically skipped on DeepSeek-V4's hash-routed layers, whose routing cannot be predicted from the previous pass. |

Each flag also has a corresponding environment variable
(`LLAMA_ARG_EXPERT_STREAMING`, `LLAMA_ARG_EXPERT_CACHE_SIZE`,
`LLAMA_ARG_EXPERT_KEEP_RECENT`, `LLAMA_ARG_EXPERT_FIXATION`,
`LLAMA_ARG_EXPERT_PREFETCH`).

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

### Example: hide SSD latency behind compute with prefetch

```bash
./build/bin/llama-cli \
  -m /path/to/DeepSeek-V4-Flash-0731-IQ1_M.gguf \
  --expert-streaming \
  --expert-prefetch \
  -p "Explain the theory of relativity." \
  -n 256
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
- Add `--expert-prefetch` alongside `--expert-streaming` whenever you can
  spare a background thread; unlike `--expert-fixation` it carries no
  accuracy cost, since it only warms the page cache ahead of time rather
  than changing routing. It is especially worthwhile on models with
  hash-routed layers (e.g. DeepSeek-V4), where fixation isn't fully
  available.

## Benchmarks

Measured on a consumer PC: 13th-gen Intel Core i7, 16GB physical RAM, NVMe
SSD, CPU-only inference (no GPU offload). Throughput is instantaneous
tokens/s measured at decode token 400.

| Model | Quant | Size (GB) | normal | expert-streaming | expert-prefetch | keep-recent 12 | fixation N=2 | fixation N=4 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B   | Q4_K_M | 19.7 | ❌ fails to launch | 7.01 | — | 5.89 | **7.28** | 5.99 ⚠️ |
| Qwen3.5-122B-A10B | Q2_K   | 42.6 | ❌ fails to launch | **2.00** | — | 1.00 | 0.35 ⚠️ | 0.53 ⚠️ |
| MiniMax-2.7       | IQ1_M  | 60.7 | ❌ fails to launch | **0.75** | — | 0.44 | 0.53 ⚠️ | 0.40 ⚠️ |
| DeepSeek-V4-Flash-0731 | IQ1_M | 80.9 GB | ❌ fails to launch | 0.39 | **0.55** | 0.55 | -¹ | -¹ |
| Qwen3.8-Flash-Next | IQ1_M | 69.4 GB | ❌ fails to launch | **0.63** | -¹ | -¹ | -¹ | -¹ |

⚠️ = a numeric throughput value was measured, but generation quality was
visually confirmed to have collapsed (repetitive/incoherent output) under
this condition — the speed number is **not** representative of usable
output. See the [technical report](#technical-report) for the full
discussion.

— = not benchmarked for this model in this round.

¹ = Expert Fixation is not supported for DeepSeek-V4-Flash-0731: the model's
hash-routed layers select experts from the input token rather than the
hidden state, so pinning the previous pass's top-k result to a fixed window
doesn't apply to them the way it does on other architectures.

**Key takeaways:**

- Expert Streaming alone turns "does not launch" into 0.39–7.01 tok/s across
  all four models, with no observed quality impact.
- `--expert-prefetch` gave a clean, accuracy-free speedup on
  DeepSeek-V4-Flash-0731, raising throughput from 0.39 tok/s
  (`--expert-streaming` alone) to **0.55 tok/s** on the 16GB-RAM test
  machine, by overlapping the next pass's expert page-ins with the current
  pass's compute.
- `--expert-keep-recent` was slower than plain `expert-streaming` in every
  test here except DeepSeek-V4-Flash-0731 — explicit eviction costs more
  than it saves compared to the OS's own page-cache management, unless you
  specifically need a RAM cap.
- `--expert-fixation` only paid off cleanly on the smallest/highest-active-parameter
  model (Qwen3.5-35B-A3B at N=2). On larger or more aggressively quantized
  models, fixation reliably degraded output quality even though the
  raw token rate looked fine. It is not applicable at all to
  DeepSeek-V4-Flash-0731's hash-routed layers.

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
- **DeepSeek-V4 hash-routed layers are not covered by Fixation or
  Prefetch.** Since those layers select experts from the input token rather
  than the hidden state, their routing can't be pinned to a prior window
  (Fixation) or predicted from the previous pass (Prefetch). Expert
  Streaming and Expert Prefetch still apply normally to the model's
  non-hash-routed layers.
- This patch is **experimental** and modifies internal loader and graph-build
  code paths in llama.cpp. Expect to need to re-port it when updating to a
  newer llama.cpp commit.

## Technical Report

A full write-up — motivation, design, implementation details (including
thread-safety and warmup handling for the fixation graph node), and the
complete experimental methodology — is included in this repository as
[`Swap-MoE_Technical_Report.pdf`](docs/Swap-MoE_Technical_Report.pdf).

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
