#pragma once

// Expert Streaming for MoE models
//
// Allows non-shared (routed) expert tensors to remain on SSD (via mmap demand
// paging) while only shared experts and dense tensors are loaded into RAM at
// startup. During inference, the OS pages in routed experts on demand.
//
// Optionally provides LRU-based explicit eviction to bound RAM usage, plus a
// prefetch hint API to warm the page cache before MoE layers.
//
// Supported architectures (matched by tensor name patterns):
//   - Qwen3.5-MoE        (blk.%d.ffn_{gate,up,down}_exps.weight)
//   - Qwen3-MoE          (same patterns)
//   - MiniMax-M2 / M2.7  (blk.%d.ffn_{gate,up,down}_exps.weight)
//   - DeepSeek-V2/V3     (same patterns; shared expert _shexp stays in RAM)
//   - Any MoE architecture that uses the standard ffn_*_exps tensor names.

#include "llama-mmap.h"
#include "llama-model-loader.h"

#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

namespace llama_expert_stream {

// Identifies whether a tensor name corresponds to a routed (non-shared) expert
// weight that should be lazily paged in from SSD.
//
// Matches patterns like:
//   blk.%d.ffn_gate_exps.weight
//   blk.%d.ffn_up_exps.weight
//   blk.%d.ffn_down_exps.weight
//   blk.%d.ffn_gate_up_exps.weight
//
// Shared expert tensors (ffn_*_shexp) and router tensors (ffn_gate_inp,
// ffn_gate_inp_shexp) are NOT routed experts and stay in RAM.
bool is_routed_expert_tensor(const std::string & name);

struct expert_region {
    // Index into llama_model_loader::mappings
    uint32_t file_idx = 0;
    // Byte offset within the mapped region
    size_t   offset  = 0;
    // Byte length of the region
    size_t   length  = 0;
    // Layer index (-1 if not layer-specific)
    int      layer   = -1;
    // Tensor name (for debugging)
    std::string name;
    // ---- Per-expert slice info (for LRU keep-recent) ----
    // For full-tensor regions: tensor_kind >= 0 and expert_id == UINT32_MAX.
    // For per-expert slice regions: tensor_kind >= 0 and expert_id < n_expert.
    int      tensor_kind = -1;   // 0=gate, 1=up, 2=down, 3=gate_up
    uint32_t expert_id   = UINT32_MAX;  // UINT32_MAX = whole tensor
};

// Identifier for a single expert slot within a layer.
// (layer, expert_id) uniquely identifies an expert weight slice across
// all routed-expert tensors (ffn_gate_exps, ffn_up_exps, ffn_down_exps,
// ffn_gate_up_exps share the same expert_id space).
struct expert_id {
    int      layer = -1;
    uint32_t id    = 0;

    bool operator==(const expert_id & o) const {
        return layer == o.layer && id == o.id;
    }
};

struct expert_id_hash {
    size_t operator()(const expert_id & e) const noexcept {
        return std::hash<int64_t>()((int64_t) e.layer * 0x100000000LL + e.id);
    }
};

// Hash key for (layer, expert_id, tensor_kind) -> region index.
struct region_key {
    int      layer;
    uint32_t expert_id;
    int      tensor_kind;

    bool operator==(const region_key & o) const {
        return layer == o.layer && expert_id == o.expert_id && tensor_kind == o.tensor_kind;
    }
};

struct region_key_hash {
    size_t operator()(const region_key & k) const noexcept {
        // Mix the three fields. tensor_kind is in {0,1,2,3}, so we can pack
        // it into the low bits.
        return ((std::hash<int64_t>()((int64_t) k.layer * 0x100000000LL + k.expert_id)) << 2)
               ^ (size_t) k.tensor_kind;
    }
};

class manager {
public:
    manager() = default;
    ~manager() = default;

    // Initialize the manager from a model loader's weight map.
    // Scans all tensor weights and records regions for routed expert tensors.
    void init(const std::map<std::string, llama_model_loader::llama_tensor_weight, llama_model_loader::weight_name_comparer> & weights);

    // Whether expert streaming is active (i.e. any expert regions found).
    bool active() const { return !regions_.empty(); }

    size_t n_regions() const { return regions_.size(); }
    size_t total_expert_bytes() const { return total_bytes_; }

    // Evict all tracked expert regions from the page cache.
    // On Linux: madvise(MADV_DONTNEED).
    // On Windows: VirtualUnlock (best-effort, requires privilege).
    // No-op on platforms without support.
    void evict_all(const llama_mmaps & mappings) const;

    // Evict a single expert region by index.
    void evict_region(const llama_mmaps & mappings, size_t idx) const;

    // Prefetch all expert regions into the page cache.
    // On Linux: madvise(MADV_WILLNEED).
    // On Windows: PrefetchVirtualMemory.
    void prefetch_all(const llama_mmaps & mappings) const;

    // Prefetch expert regions belonging to a specific layer.
    void prefetch_layer(const llama_mmaps & mappings, int layer) const;

    // Evict experts for layers > layer (keep layers <= layer paged in).
    // Useful for sequential layer processing where future layers' experts
    // can be evicted to make room for the current layer.
    void evict_layers_after(const llama_mmaps & mappings, int layer) const;

    // Periodic LRU maintenance by byte budget. Call after each token
    // generation to evict experts that haven't been recently accessed.
    //
    // When `cache_target_bytes` > 0, evicts expert regions until estimated
    // resident expert bytes <= cache_target_bytes. Eviction order is
    // descending layer index (i.e. evict experts from later layers first,
    // which are less likely to be reused in sequential decoding).
    void lru_maintain(const llama_mmaps & mappings, size_t cache_target_bytes) const;

    // ---- LRU keep-recent API ------------------------------------------------
    //
    // Tracks recently-used (layer, expert_id) pairs and explicitly evicts
    // the pages of experts that fall outside the keep-recent window.
    //
    // Configured via set_keep_recent(n). n = 0 disables (no extra caching
    // beyond OS page cache); n > 0 keeps the n most-recently-used experts
    // per layer resident in RAM.

    // Set the keep-recent window size (in experts per layer).
    // n = 0 disables the LRU keep-recent feature (default).
    void set_keep_recent(size_t n) { keep_recent_ = n; }
    size_t keep_recent() const { return keep_recent_; }

    // Notify the manager that a set of experts was used in the most recent
    // forward pass for a given layer. The manager updates its LRU state.
    // Thread-safe.
    void notify_used(int layer, const uint32_t * expert_ids, size_t n);

    // Run LRU maintenance: evict pages for experts no longer in the
    // keep-recent window. Call once after a full forward pass.
    // Thread-safe. Returns the number of distinct experts evicted.
    size_t lru_evict_stale(const llama_mmaps & mappings);

    // Print a one-line summary of tracked regions.
    void print_summary() const;

    // For debugging: number of experts currently marked resident.
    size_t lru_resident_count() const;

private:
    std::vector<expert_region> regions_;
    // Index: layer -> list of indices into regions_
    std::unordered_map<int, std::vector<size_t>> layer_index_;
    // Index: (layer, expert_id, tensor_kind) -> region index. O(1) lookup.
    std::unordered_map<region_key, size_t, region_key_hash> region_index_;
    size_t total_bytes_ = 0;

    // ---- LRU state ---------------------------------------------------------
    size_t keep_recent_ = 0;

    // Per-layer LRU list: front = most recently used, back = least recently.
    // We store (layer, expert_id) keyed by layer for fast access.
    // Map: layer -> list of expert_ids (MRU at front)
    std::unordered_map<int, std::list<uint32_t>> lru_per_layer_;
    mutable std::mutex lru_mtx_;

    // Per-layer resident expert_id set: tracks which experts we believe are
    // currently paged into RAM. Updated by lru_evict_stale. Used to skip
    // redundant MADV_DONTNEED syscalls for already-evicted experts.
    std::unordered_map<int, std::unordered_set<uint32_t>> resident_per_layer_;

    // Find the region index for a given (layer, expert_id, tensor_kind).
    // tensor_kind: 0=gate_exps, 1=up_exps, 2=down_exps, 3=gate_up_exps
    // Returns SIZE_MAX if not found. O(1) via region_index_.
    size_t find_region(int layer, uint32_t expert_id, int tensor_kind) const;
};

// ---------------------------------------------------------------------------
// Expert fixation (v7 redesign)
//
// v6 implemented "--expert-fixation N" by registering a
// ggml_backend_sched eval callback (ggml_backend_sched_set_eval_callback).
// This turned out to have two serious performance problems, found by
// reading ggml/src/ggml-backend.cpp's split-compute loop:
//
//   1. Whenever ANY callback is registered on the scheduler, the scheduler
//      abandons its normal "one async compute call per split" fast path and
//      instead walks every single node of the graph, calling the callback
//      with ask=true for each one to decide whether to break the graph
//      there. Each node where the callback answers "yes" forces a
//      ggml_graph_view + a separate ggml_backend_graph_compute_async call +
//      an *immediate* ggml_backend_synchronize call, right there in the
//      middle of what would otherwise be one uninterrupted graph dispatch.
//      For a 62-layer MoE model this fragments a single-token forward pass
//      into ~62 separate dispatches, each with its own thread-pool
//      wake/synchronize overhead -- easily outweighing the SSD I/O it was
//      meant to save.
//   2. The callback was registered once fixation was *configured* and never
//      unregistered, even after the fixation window (N forward passes)
//      elapsed. So the per-node ask() traversal overhead described above
//      (though degenerating to "answer false for everything") continued for
//      every token of the entire session, not just the first N.
//
// v7 removes the eval-callback mechanism entirely. Instead, the top-k
// override is implemented as an ordinary ggml graph node
// (GGML_OP_MAP_CUSTOM1, see fixation_apply()) inserted directly after the
// router's argsort_top_k op in llm_graph_context::build_moe_ffn(). This
// node participates in the model's normal single-shot graph dispatch like
// any other op:
//   - No scheduler callback is ever registered, so graph batching and
//     "graph reuse" across tokens are both fully preserved.
//   - The node is present in the graph whenever fixation is *configured*
//     (so the graph's shape/topology never changes based on the live
//     countdown), but when the countdown has reached zero the node is just
//     a small memcpy of the routed-expert-id tensor (a few dozen bytes for
//     a typical n_expert_used) executed inline -- no extra synchronization,
//     no extra dispatch, regardless of how many tokens follow.

// Shared runtime state for expert fixation. Owned by llama_model::impl and
// referenced (read/write) from graph nodes built by build_moe_ffn().
struct fixation_state {
    // Number of remaining forward passes for which top-k should be pinned.
    // <= 0 means inactive. Decremented once per (non-warmup) token via
    // llama_model::expert_stream_tick_fixation() -- NOT once per layer.
    std::atomic<int32_t> remaining{0};

    // True while a warmup forward pass is in flight. The fixation node must
    // not capture (or override with) data from a warmup pass, since warmup
    // uses a placeholder token rather than real prompt/generation content.
    std::atomic<bool> warmup{false};

    // Whether --expert-fixation was configured with N > 0 at load time.
    // Unlike `remaining`, this never changes after model load: it controls
    // whether the fixation graph node is inserted at all, so the graph
    // topology stays constant for the lifetime of the context.
    bool configured = false;

    // Per-layer cache of the pinned expert-id selection, captured from the
    // first real (non-warmup) forward pass of the fixation window.
    // Each layer's entry is touched by exactly one graph node running with
    // n_tasks=1, and layers execute strictly sequentially within a single
    // graph compute, so no locking is required.
    std::vector<std::vector<int32_t>> cache;
    std::vector<uint8_t>              cache_valid;

    // Reset countdown/cache state. Called once at load time (n = configured
    // --expert-fixation value, possibly 0) and does not need to be called
    // again afterwards.
    void configure(int32_t n, size_t n_layer) {
        configured = n > 0;
        remaining.store(n > 0 ? n : 0, std::memory_order_relaxed);
        warmup.store(false, std::memory_order_relaxed);
        cache.assign(n_layer, {});
        cache_valid.assign(n_layer, 0);
    }
};

// Per-graph-node arguments baked into the ggml_map_custom1 userdata
// pointer. One instance per layer, owned for the lifetime of the model
// (stored in a fixed-size, never-reallocated vector) so the raw pointer
// stays valid across graph rebuilds and graph reuse.
struct fixation_node_args {
    fixation_state * state = nullptr;
    int              il    = -1;
};

// ggml_custom1_op_t implementation. Copies the router's real top-k
// selection from `a` into `dst`, then -- only while fixation is active and
// not in a warmup pass -- either captures it (first pinned pass for this
// layer) or overwrites `dst` with the previously-captured selection
// (subsequent pinned passes), forcing the same routed experts to be reused.
// Runs with n_tasks=1 (see build_moe_ffn), so ith/nth are unused.
void fixation_apply(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

} // namespace llama_expert_stream
