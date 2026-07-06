#include "llama-expert-stream.h"

#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-mmap.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>

#ifdef __has_include
#  if __has_include(<unistd.h>)
#    include <unistd.h>
#    include <sys/mman.h>
#  endif
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace llama_expert_stream {

// Patterns that identify routed (non-shared) expert weight tensors.
// We check for these substrings in the tensor name.
static bool contains_substr(const std::string & s, const char * sub) {
    return strstr(s.c_str(), sub) != nullptr;
}

bool is_routed_expert_tensor(const std::string & name) {
    // Shared expert tensors stay in RAM. The shared-expert check MUST run
    // before the routed-expert check because some shared expert tensor
    // names (e.g. "ffn_gate_inp_shexp") contain substrings that overlap
    // with routed-expert patterns ("ffn_gate_inp" matches the router
    // pattern, "ffn_gate_shexp" contains "ffn_gate_*" if we were lax).
    //
    // We check the "_shexp" suffix patterns explicitly to avoid false
    // positives on hypothetical names like "ffn_gate_exps_shexp".
    if (contains_substr(name, "ffn_gate_inp_shexp") ||
        contains_substr(name, "ffn_gate_shexp")     ||
        contains_substr(name, "ffn_up_shexp")       ||
        contains_substr(name, "ffn_down_shexp")     ||
        // Defensive: any name containing "_shexp" is a shared expert.
        contains_substr(name, "_shexp")) {
        return false;
    }
    // Router (gate_inp without _shexp suffix) stays in RAM — it's tiny and hot.
    if (contains_substr(name, "ffn_gate_inp")) {
        return false;
    }
    // Routed expert weight tensors
    if (contains_substr(name, "ffn_gate_exps")  ||
        contains_substr(name, "ffn_up_exps")    ||
        contains_substr(name, "ffn_down_exps")  ||
        contains_substr(name, "ffn_gate_up_exps")) {
        return true;
    }
    return false;
}

// Classify a routed-expert tensor name into a kind tag.
//   0 = ffn_gate_exps
//   1 = ffn_up_exps
//   2 = ffn_down_exps
//   3 = ffn_gate_up_exps (merged gate+up)
//   -1 = unknown (shouldn't happen for routed-expert tensors)
static int tensor_kind_from_name(const std::string & name) {
    // IMPORTANT: check gate_up BEFORE gate, because "ffn_gate_up_exps"
    // contains the substring "ffn_gate_exps" is FALSE (since the substring
    // would actually be "ffn_gate_exps" but the name is "ffn_gate_up_exps").
    // However, "ffn_gate_exps" is a substring of "ffn_gate_exps_foo" not
    // "ffn_gate_up_exps". So the order below is safe. But to be defensive
    // we check gate_up first anyway.
    if (contains_substr(name, "ffn_gate_up_exps")) return 3;
    if (contains_substr(name, "ffn_gate_exps"))   return 0;
    if (contains_substr(name, "ffn_up_exps"))     return 1;
    if (contains_substr(name, "ffn_down_exps"))   return 2;
    return -1;
}

static const char * tensor_kind_name(int kind) {
    switch (kind) {
        case 0: return "gate";
        case 1: return "up";
        case 2: return "down";
        case 3: return "gate_up";
        default: return "?";
    }
}

void manager::init(const std::map<std::string, llama_model_loader::llama_tensor_weight, llama_model_loader::weight_name_comparer> & weights) {
    regions_.clear();
    layer_index_.clear();
    region_index_.clear();
    total_bytes_ = 0;
    lru_per_layer_.clear();
    resident_per_layer_.clear();

    for (const auto & [name, weight] : weights) {
        if (!is_routed_expert_tensor(name)) {
            continue;
        }

        int kind = tensor_kind_from_name(name);
        if (kind < 0) {
            continue;
        }

        // Parse layer index from "blk.%d."
        int layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) != 1) {
            layer = -1;
        }

        const ggml_tensor * t = weight.tensor;
        // For routed-expert tensors the last dimension (ne[2] for 3-D, or
        // ne[1] for 2-D) is n_expert. Compute per-expert byte stride.
        // ne[0] = fastest-changing dim.
        // For a 3-D tensor (n_embd, n_ff, n_expert): total bytes = nbytes,
        // per-expert bytes = nbytes / n_expert.
        // For a 2-D tensor we wouldn't be here (no expert dimension).
        // We compute n_expert from the leading dims.
        size_t total_nbytes = ggml_nbytes(t);
        // Determine n_expert from the tensor's trailing dims.
        // ggml tensors are stored with ne[0] as the contiguous dimension.
        // For routed-expert tensors, ne[2] (or ne[1] for 2-D) is n_expert.
        uint32_t n_expert = 1;
        if (t->ne[2] > 1) {
            n_expert = (uint32_t) t->ne[2];
        } else if (t->ne[1] > 1) {
            n_expert = (uint32_t) t->ne[1];
        }
        if (n_expert == 0) {
            n_expert = 1;
        }
        // Per-expert byte stride. For quantized tensors the row stride is
        // nb[1] * (ne[1]) per expert; easier: total / n_expert, since each
        // expert slice is contiguous in memory (ggml stores the expert dim
        // as the outermost slowest-changing dim).
        size_t per_expert_bytes = total_nbytes / n_expert;

        for (uint32_t e = 0; e < n_expert; ++e) {
            expert_region r;
            r.file_idx     = weight.idx;
            r.offset       = weight.offs + e * per_expert_bytes;
            r.length       = per_expert_bytes;
            r.layer        = layer;
            r.name         = name;
            r.tensor_kind  = kind;
            r.expert_id    = e;

            total_bytes_ += r.length;
            regions_.push_back(std::move(r));
        }
    }

    // Sort regions by (file_idx, layer, offset) for nicer eviction patterns.
    // The region_index_ is rebuilt AFTER the sort so its indices are stable.
    std::sort(regions_.begin(), regions_.end(),
        [](const expert_region & a, const expert_region & b) {
            if (a.file_idx != b.file_idx) return a.file_idx < b.file_idx;
            if (a.layer    != b.layer)    return a.layer    < b.layer;
            return a.offset < b.offset;
        });

    // Rebuild layer_index_ and region_index_ after sort.
    layer_index_.clear();
    region_index_.clear();
    region_index_.reserve(regions_.size());
    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto & r = regions_[i];
        layer_index_[r.layer].push_back(i);
        region_key key{r.layer, r.expert_id, r.tensor_kind};
        region_index_[key] = i;
    }
}

// Platform-specific page-level operations on a contiguous range within a
// mapping. These are best-effort — failures are logged but do not abort.
static void advise_range_willneed(void * base, size_t len) {
    if (base == nullptr || len == 0) return;
#if defined(_POSIX_MAPPED_FILES) && defined(POSIX_MADV_WILLNEED)
    int rc = posix_madvise(base, len, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        LLAMA_LOG_DEBUG("expert_stream: posix_madvise(WILLNEED) failed: %s\n", strerror(rc));
    }
#elif defined(_WIN32) && _WIN32_WINNT >= 0x602
    BOOL (WINAPI *pPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG) = nullptr;
    HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
    if (hKernel32) {
        pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))
            GetProcAddress(hKernel32, "PrefetchVirtualMemory");
    }
    if (pPrefetchVirtualMemory) {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = base;
        range.NumberOfBytes  = len;
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
            LLAMA_LOG_DEBUG("expert_stream: PrefetchVirtualMemory failed: %lu\n", GetLastError());
        }
    }
#else
    GGML_UNUSED(base); GGML_UNUSED(len);
#endif
}

static void advise_range_dontneed(void * base, size_t len) {
    if (base == nullptr || len == 0) return;
#if defined(_POSIX_MAPPED_FILES) && defined(POSIX_MADV_DONTNEED)
    // Align to page size
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    uintptr_t addr = (uintptr_t) base;
    uintptr_t end  = addr + len;
    uintptr_t a0   = (addr + ps - 1) & ~((uintptr_t) ps - 1);
    uintptr_t a1   = end & ~((uintptr_t) ps - 1);
    if (a1 <= a0) return;
    int rc = posix_madvise((void *) a0, a1 - a0, POSIX_MADV_DONTNEED);
    if (rc != 0) {
        LLAMA_LOG_DEBUG("expert_stream: posix_madvise(DONTNEED) failed: %s\n", strerror(rc));
    }
#elif defined(_WIN32)
    // VirtualUnlock evicts pages from the working set.
    // It requires the process to hold SE_DEBUG_NAME or similar privilege,
    // and may fail silently otherwise — that's acceptable, the OS will
    // still page out under memory pressure.
    if (!VirtualUnlock(base, len)) {
        DWORD err = GetLastError();
        // Error 0x9C (ERROR_NOT_LOCKED) is normal: pages already not locked.
        if (err != 158 && err != 0x9C) {
            LLAMA_LOG_DEBUG("expert_stream: VirtualUnlock failed: %lu\n", err);
        }
    }
#else
    GGML_UNUSED(base); GGML_UNUSED(len);
#endif
}

static void * region_addr(const llama_mmaps & mappings, const expert_region & r) {
    if (r.file_idx >= mappings.size()) return nullptr;
    return (uint8_t *) mappings[r.file_idx]->addr() + r.offset;
}

void manager::evict_all(const llama_mmaps & mappings) const {
    for (const auto & r : regions_) {
        advise_range_dontneed(region_addr(mappings, r), r.length);
    }
}

void manager::evict_region(const llama_mmaps & mappings, size_t idx) const {
    if (idx >= regions_.size()) return;
    advise_range_dontneed(region_addr(mappings, regions_[idx]), regions_[idx].length);
}

void manager::prefetch_all(const llama_mmaps & mappings) const {
    for (const auto & r : regions_) {
        advise_range_willneed(region_addr(mappings, r), r.length);
    }
}

void manager::prefetch_layer(const llama_mmaps & mappings, int layer) const {
    auto it = layer_index_.find(layer);
    if (it == layer_index_.end()) return;
    for (size_t idx : it->second) {
        advise_range_willneed(region_addr(mappings, regions_[idx]), regions_[idx].length);
    }
}

void manager::evict_layers_after(const llama_mmaps & mappings, int layer) const {
    for (const auto & r : regions_) {
        if (r.layer > layer) {
            advise_range_dontneed(region_addr(mappings, r), r.length);
        }
    }
}

void manager::lru_maintain(const llama_mmaps & mappings, size_t cache_target_bytes) const {
    if (regions_.empty() || cache_target_bytes == 0) return;

    // Evict from highest layer index downward (later layers are least
    // likely to be reused in sequential decoding).
    size_t resident_estimate = total_bytes_;
    if (resident_estimate <= cache_target_bytes) return;

    // Build a list of regions sorted by descending layer.
    std::vector<size_t> order;
    order.reserve(regions_.size());
    for (size_t i = 0; i < regions_.size(); ++i) order.push_back(i);
    std::sort(order.begin(), order.end(),
        [this](size_t a, size_t b) {
            return regions_[a].layer > regions_[b].layer;
        });

    for (size_t idx : order) {
        if (resident_estimate <= cache_target_bytes) break;
        advise_range_dontneed(region_addr(mappings, regions_[idx]), regions_[idx].length);
        resident_estimate -= regions_[idx].length;
    }
}

void manager::print_summary() const {
    if (regions_.empty()) {
        LLAMA_LOG_INFO("expert_stream: no routed expert regions tracked\n");
        return;
    }
    LLAMA_LOG_INFO("expert_stream: tracking %zu routed expert slices, total %.2f GiB\n",
        regions_.size(), (double) total_bytes_ / 1024.0 / 1024.0 / 1024.0);
}

// ---- LRU keep-recent implementation ---------------------------------------

size_t manager::find_region(int layer, uint32_t expert_id, int tensor_kind) const {
    // O(1) hash lookup via region_index_.
    region_key key{layer, expert_id, tensor_kind};
    auto it = region_index_.find(key);
    if (it == region_index_.end()) {
        return SIZE_MAX;
    }
    return it->second;
}

void manager::notify_used(int layer, const uint32_t * expert_ids, size_t n) {
    if (keep_recent_ == 0 || n == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(lru_mtx_);
    auto & lst = lru_per_layer_[layer];
    for (size_t i = 0; i < n; ++i) {
        uint32_t eid = expert_ids[i];
        // Remove existing entry (if any) so we can re-insert at front.
        // lst.remove is O(window size); acceptable since the window is
        // typically small (n_expert_used per token).
        lst.remove(eid);
        lst.push_front(eid);
    }
    // No truncation here — we truncate in lru_evict_stale to batch the work.
}

size_t manager::lru_evict_stale(const llama_mmaps & mappings) {
    if (keep_recent_ == 0 || regions_.empty()) {
        return 0;
    }

    // Step 1: Snapshot the per-layer LRU lists under the lock.
    std::unordered_map<int, std::list<uint32_t>> snapshot;
    {
        std::lock_guard<std::mutex> lk(lru_mtx_);
        snapshot = lru_per_layer_;
    }

    // Step 2: For each layer that has LRU state, compute the new resident
    // set (first keep_recent_ entries in the LRU list).
    //
    // Optimization: instead of scanning all regions (which can be 47K+ for
    // MiniMax-M2), we iterate two sets per layer:
    //   - new_resident[layer] = first keep_recent_ of snapshot[layer]
    //   - to_evict[layer]     = (snapshot[layer] beyond keep_recent_) UNION
    //                          (old resident_per_layer_[layer] not in new_resident)
    //
    // For each expert_id in to_evict[layer], we evict all 3-4 tensor kinds
    // via region_index_ (O(1) hash lookup per kind).
    //
    // This reduces per-token work from O(n_regions * keep_recent) to
    // O(n_active_layers * (keep_recent + n_evicted_experts * 4)).
    //
    // IMPORTANT: We must evict experts that are in the LRU list but BEYOND
    // the keep_recent_ window, even if resident_per_layer_ never tracked
    // them. Otherwise, on the first few tokens after startup (when
    // resident_per_layer_ is empty), experts that were used and paged in
    // but fell out of the window would never be evicted, causing unbounded
    // RAM growth.

    size_t evicted_experts = 0;

    // First, build new_resident AND to_evict_extra for every layer present
    // in the snapshot.
    std::unordered_map<int, std::unordered_set<uint32_t>> new_resident;
    std::unordered_map<int, std::unordered_set<uint32_t>> to_evict_extra;
    new_resident.reserve(snapshot.size());
    for (auto & [layer, lst] : snapshot) {
        auto & s = new_resident[layer];
        auto & extra = to_evict_extra[layer];
        size_t cnt = 0;
        for (auto it = lst.begin(); it != lst.end(); ++it) {
            if (cnt < keep_recent_) {
                s.insert(*it);
                ++cnt;
            } else {
                // Beyond the keep window — candidate for eviction.
                extra.insert(*it);
            }
        }
    }

    // For each layer we have previously tracked as resident, evict experts
    // that are no longer in the new resident set.
    for (auto & [layer, old_set] : resident_per_layer_) {
        auto new_it = new_resident.find(layer);
        const std::unordered_set<uint32_t> empty_set;
        const auto & new_set = (new_it != new_resident.end()) ? new_it->second : empty_set;

        for (uint32_t eid : old_set) {
            if (new_set.find(eid) != new_set.end()) {
                continue;  // Still resident — no eviction needed.
            }
            // Evict all tensor kinds for this (layer, expert_id).
            // Typical MoE has 3 kinds: gate, up, down. Some merge gate+up
            // into gate_up (1 kind). We check all 4.
            for (int kind = 0; kind < 4; ++kind) {
                size_t idx = find_region(layer, eid, kind);
                if (idx == SIZE_MAX) continue;
                const auto & r = regions_[idx];
                advise_range_dontneed(region_addr(mappings, r), r.length);
            }
            ++evicted_experts;
        }
    }

    // Now evict experts that are in the LRU list beyond the keep window but
    // were not previously tracked in resident_per_layer_ (e.g. on the first
    // few tokens after startup when resident_per_layer_ was still empty).
    // We skip experts already evicted above by checking new_resident.
    for (auto & [layer, extra] : to_evict_extra) {
        auto new_it = new_resident.find(layer);
        const std::unordered_set<uint32_t> empty_set;
        const auto & new_set = (new_it != new_resident.end()) ? new_it->second : empty_set;

        for (uint32_t eid : extra) {
            if (new_set.find(eid) != new_set.end()) {
                continue;  // Should not happen, but defensive.
            }
            for (int kind = 0; kind < 4; ++kind) {
                size_t idx = find_region(layer, eid, kind);
                if (idx == SIZE_MAX) continue;
                const auto & r = regions_[idx];
                advise_range_dontneed(region_addr(mappings, r), r.length);
            }
            ++evicted_experts;
        }
    }

    // Step 3: Update resident_per_layer_ to reflect the new state.
    resident_per_layer_ = std::move(new_resident);

    return evicted_experts;
}

size_t manager::lru_resident_count() const {
    if (keep_recent_ == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(lru_mtx_);
    size_t total = 0;
    for (const auto & [layer, lst] : lru_per_layer_) {
        total += std::min(keep_recent_, lst.size());
    }
    return total;
}

// ---- Expert fixation (v7 graph-node based) --------------------------------

void fixation_apply(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);

    // IMPORTANT: ggml's CPU thread dispatcher invokes this function once per
    // pool thread for every node (ggml_graph_compute_thread() in
    // ggml-cpu.c calls ggml_compute_forward() unconditionally for all
    // `-t N` threads; it does NOT gate calls based on the n_tasks value we
    // pass to ggml_map_custom1()). n_tasks only affects work-buffer sizing,
    // not which threads actually get invoked. So this callback -- like any
    // ggml_map_custom*/GGML_OP_MAP_CUSTOM* callback that wants to run only
    // once -- must self-gate on `ith`.
    //
    // Without this guard, every one of the `-t N` compute threads ran the
    // capture/override logic below concurrently, all mutating the same
    // (non-thread-safe) std::vector cache at once -- corrupting the heap
    // (observed on Windows as STATUS_HEAP_CORRUPTION / 0xc0000374 the
    // moment --expert-fixation was combined with more than one CPU thread).
    if (ith != 0) {
        return;
    }

    // dst is a fresh tensor (ggml_dup_tensor), so it always has its own
    // buffer distinct from `a`. Start from a faithful copy of the router's
    // real selection; every early-return path below leaves dst holding
    // exactly that.
    const size_t nbytes = ggml_nbytes(a);
    memcpy(dst->data, a->data, nbytes);

    auto * args = static_cast<fixation_node_args *>(userdata);
    if (args == nullptr || args->state == nullptr || args->il < 0) {
        return;
    }

    fixation_state * st = args->state;

    // Never capture or override during a warmup forward pass -- warmup uses
    // a placeholder token, not real prompt/generation content.
    if (st->warmup.load(std::memory_order_relaxed)) {
        return;
    }

    const int32_t remaining = st->remaining.load(std::memory_order_relaxed);
    if (remaining <= 0) {
        // Fixation inactive (window not started yet, or already elapsed).
        // dst already holds the router's real selection -- nothing to do.
        return;
    }

    if (a->type != GGML_TYPE_I32) {
        return;
    }

    const size_t count = ggml_nelements(a);
    if (count == 0 || (size_t) args->il >= st->cache.size()) {
        return;
    }

    std::vector<int32_t> & cache = st->cache[args->il];
    uint8_t               & valid = st->cache_valid[args->il];

    if (!valid || cache.size() != count) {
        // First pinned pass for this layer, or the ubatch shape changed
        // since the last capture (e.g. prefill vs. single-token decode).
        // Record the router's real selection; dst already holds it.
        const int32_t * src = (const int32_t *) a->data;
        cache.assign(src, src + count);
        valid = 1;
        return;
    }

    // Subsequent pinned pass with matching shape: force the pinned
    // expert IDs so the same routed experts are reused (and their pages
    // stay resident instead of triggering fresh SSD reads).
    memcpy(dst->data, cache.data(), count * sizeof(int32_t));
}

} // namespace llama_expert_stream
