#include "module/decode_kv_write.h"

#include "model/util.h"

namespace llm_system {

DecodeKVWrite::DecodeKVWrite(std::string& prefix, std::string& name,
                             const ModelConfig& model_config,
                             int pp_rank, int layer_start, int layer_end,
                             std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),  // graph_execution=true
      model_config(model_config),
      pp_rank(pp_rank), layer_start(layer_start), layer_end(layer_end) {
  // Register placeholder output tensor
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("kv_write_output", shape, "act",
                                       device, model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr DecodeKVWrite::forward(const Tensor::Ptr input,
                                    BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output = get_activation("kv_write_output", input->shape);

  if (!device->perform_execution) return output;

  // Compute stage-local KV bulk write bytes for sequences that haven't
  // written KV to this device yet. Each TP device writes its own shard
  // independently — tracked per device_total_rank, not per stage.
  int my_rank = device->device_total_rank;
  long long total_write_bytes = 0;

  for (const auto& seq : sequences_metadata->get_seq_ref()) {
    if (seq->num_process_token == 0) continue;

    // Device-level KV write completion check.
    // Same Sequence is shared by TP devices in same DP replica,
    // but each device inserts its own rank — no interference.
    if (seq->prompt_kv_written_devices.count(my_rank)) continue;

    // Use current_len instead of input_len: seeded steady-state sequences
    // have current_len > input_len, so historical decode KV is also written.
    // For new (non-seeded) sequences, current_len == input_len → same as before.
    int full_len = seq->current_len;
    int reused_len = seq->prompt_reused_len;

    // Standard KV cache: per-layer, TP-sharded.
    // K + V = 2 tensors per layer, each: tokens * head_dim * (num_kv_heads / ne_tp_dg).
    // TODO: Add MLA/compressed_kv/absorb branch when needed.
    for (int l = layer_start; l < layer_end; l++) {
      int eff_full = effective_kv_len(model_config, l, full_len);
      int eff_reused = effective_reused_kv_len(model_config, l, full_len, reused_len);
      int new_kv_tokens = eff_full - eff_reused;
      if (new_kv_tokens <= 0) continue;

      long long layer_bytes = 2LL * new_kv_tokens * model_config.head_dim
          * model_config.num_kv_heads / model_config.ne_tp_dg
          * model_config.precision_byte;
      total_write_bytes += layer_bytes;
    }

    // Mark this device as having written KV for this sequence
    seq->prompt_kv_written_devices.insert(my_rank);
  }

  if (total_write_bytes > 0) {
    // Accumulate KV write for PEC tracking (HBF only)
    if (device->config.gpu_gen.rfind("HBF", 0) == 0) {
      device->cumulative_kv_write_bytes += total_write_bytes;
    }

    // v1: time-only latency/stall model. Only device_time is updated.
    // TODO: Reflect write_energy and all_write_energy in StatusBoard
    //       so that getTotalEnergy() and per-iteration energy stats are accurate.
    // TODO: Integrate with ramulator path (device->dram_interface) for
    //       cycle-accurate HBM write modeling when use_ramulator=true.
    time_ns write_time = (double)total_write_bytes
        / device->config.memory_write_bandwidth * 1e9;
    device->status.device_time += write_time;
  }

  return output;
}

}  // namespace llm_system
