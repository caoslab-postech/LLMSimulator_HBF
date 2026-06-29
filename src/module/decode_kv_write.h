#pragma once
#include <string>
#include <vector>

#include "model/model_config.h"
#include "module/module.h"
#include "scheduler/sequence.h"

namespace llm_system {

// Leaf module: stage-local KV cache bulk write for disagg decode-only mode.
// On each sequence's first visit to this DEVICE, writes the TP-local KV shard
// to HBM. Subsequent visits are no-ops (tracked via prompt_kv_written_devices).
//
// v1: time-only model (device_time += write_time).
// TODO: Reflect write_energy, all_write_energy in StatusBoard.
// TODO: Integrate with ramulator path for cycle-accurate HBM write.
// TODO: Add MLA/compressed_kv/absorb branch.
class DecodeKVWrite : public Module {
 public:
  using Ptr = std::shared_ptr<DecodeKVWrite>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  int pp_rank, int layer_start, int layer_end,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new DecodeKVWrite(prefix, name, model_config,
                  pp_rank, layer_start, layer_end, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  DecodeKVWrite(std::string& prefix, std::string& name,
                const ModelConfig& model_config,
                int pp_rank, int layer_start, int layer_end,
                std::vector<int> device_list, Device::Ptr device);
  DecodeKVWrite() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
  int pp_rank;
  int layer_start;
  int layer_end;
};

}  // namespace llm_system
