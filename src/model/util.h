#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "model/model_config.h"

namespace llm_system {

// Chunked attention helpers: full attention layers keep full KV, others clamp to window
inline bool keep_full_kv(const ModelConfig& cfg, int layer_idx) {
  if (!cfg.use_chunked_attention) return true;
  return (layer_idx % cfg.chunked_attention_full_stride) == 0;
}

inline int effective_kv_len(const ModelConfig& cfg, int layer_idx, int full_len) {
  if (keep_full_kv(cfg, layer_idx)) return full_len;
  return std::min(full_len, cfg.chunked_attention_window);
}

inline int effective_reused_kv_len(const ModelConfig& cfg, int layer_idx,
                                   int full_len, int reused_len) {
  if (keep_full_kv(cfg, layer_idx)) {
    return std::min(reused_len, full_len);
  }

  int window_start = std::max(0, full_len - cfg.chunked_attention_window);
  int reused_prefix_end = std::min(reused_len, full_len);
  return std::max(0, reused_prefix_end - window_start);
}

// Overload for use in hardware execution with LayerInfo fields
inline int effective_kv_len(int decoder_idx, int chunked_attention_full_stride,
                            int chunked_attention_window, int full_len) {
  if ((decoder_idx % chunked_attention_full_stride) == 0) return full_len;
  return std::min(full_len, chunked_attention_window);
}

void inline set_device_list(std::vector<int> &device_list, int device_offset,
                            int num_device) {
  device_list.resize(0);

  for (int device = device_offset; device < device_offset + num_device;
       device++) {
    device_list.push_back(device);
  }
}
// --- Pipeline Parallelism helpers ---

// Rank decomposition for TP(innermost) -> PP -> DP(outermost) layout.
struct RankInfo {
    int tp_rank;          // = global_rank % ne_tp_dg
    int pp_rank;          // = (global_rank / ne_tp_dg) % pp_dg
    int dp_rank;          // = global_rank / (ne_tp_dg * pp_dg)
    int tp_group_offset;  // = floor(global_rank / ne_tp_dg) * ne_tp_dg
};

// Decompose a global device rank into (tp, pp, dp) coordinates.
inline RankInfo decompose_rank(int global_rank, int ne_tp_dg, int pp_dg) {
    RankInfo r;
    r.tp_rank         = global_rank % ne_tp_dg;
    r.pp_rank         = (global_rank / ne_tp_dg) % pp_dg;
    r.dp_rank         = global_rank / (ne_tp_dg * pp_dg);
    r.tp_group_offset = (global_rank / ne_tp_dg) * ne_tp_dg;
    return r;
}

// Compute the global rank of the corresponding device in an adjacent PP stage.
// direction: +1 = next stage, -1 = previous stage.
inline int get_pp_peer(int global_rank, int ne_tp_dg, int direction) {
    return global_rank + direction * ne_tp_dg;
}

// Compute per-stage layer counts for pipeline parallelism.
// Policy: chunked attention or MoE require even split; dense non-MoE allows
// uneven split with remainder distributed to center stages first.
inline std::vector<int> get_stage_layer_counts(const ModelConfig& cfg) {
    int pp = cfg.pp_dg;
    int base = cfg.num_layers / pp;
    int rem  = cfg.num_layers % pp;

    // Chunked attention: must divide evenly, and each stage must contain
    // a whole number of full_stride groups.
    if (cfg.use_chunked_attention) {
        assertTrue(rem == 0,
                   "chunked attention requires num_layers divisible by pp_dg");
        assertTrue(base % cfg.chunked_attention_full_stride == 0,
                   "layers_per_stage must be divisible by chunked_attention_full_stride");
        return std::vector<int>(pp, base);
    }

    // MoE models: require even split for safety.
    if (cfg.expert_freq != 0) {
        assertTrue(rem == 0,
                   "MoE models require num_layers divisible by pp_dg");
        return std::vector<int>(pp, base);
    }

    // Dense non-MoE: uneven split allowed.
    // Distribute remainder to center stages first (center-outward priority).
    std::vector<int> counts(pp, base);
    std::vector<int> priority(pp);
    for (int i = 0; i < pp; i++) priority[i] = i;
    std::sort(priority.begin(), priority.end(), [&](int a, int b) {
        double ca = std::abs(a - (pp - 1) / 2.0);
        double cb = std::abs(b - (pp - 1) / 2.0);
        return ca < cb;
    });
    for (int i = 0; i < rem; i++) counts[priority[i]]++;
    return counts;
}

// Return [start_layer, end_layer) for the given pp_rank.
inline std::pair<int, int> get_stage_layer_range(const ModelConfig& cfg, int pp_rank) {
    auto counts = get_stage_layer_counts(cfg);
    int start = 0;
    for (int i = 0; i < pp_rank; i++) start += counts[i];
    return {start, start + counts[pp_rank]};
}

// All devices belonging to the same PP stage across all DP replicas.
// Non-contiguous when dp_degree > 1.
// Ordered by DP rank: dp0's TP group first, dp1's next, etc.
// This ordering guarantees: chunk index i (i-th ne_tp_dg block) = dp_rank i.
// Example: pp_rank=0, ne_tp=8, pp=2, dp=2 → {0..7, 16..23}
inline std::vector<int> get_stage_device_list(int pp_rank, int ne_tp_dg,
                                              int pp_dg, int dp_degree) {
    std::vector<int> devices;
    for (int dp = 0; dp < dp_degree; ++dp) {
        int base = dp * (pp_dg * ne_tp_dg) + pp_rank * ne_tp_dg;
        for (int t = 0; t < ne_tp_dg; ++t)
            devices.push_back(base + t);
    }
    return devices;
}

// Map a global device rank to its index within a (possibly non-contiguous) device list.
// Example: device_list={0..7, 16..23}, global_rank=16 → returns 8
inline int get_local_rank_in_list(const std::vector<int>& device_list,
                                  int global_rank) {
    for (int i = 0; i < (int)device_list.size(); ++i)
        if (device_list[i] == global_rank) return i;
    fail("get_local_rank_in_list: rank " + std::to_string(global_rank) + " not in device_list");
    return -1;
}

}  // namespace llm_system
