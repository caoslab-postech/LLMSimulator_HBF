#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "module/parallel.h"        // ColumnParallelLinear
#include "module/communication.h"   // AllGather
#include "scheduler/sequence.h"

namespace llm_system {

class LmHead : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<LmHead>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new LmHead(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  LmHead(std::string& prefix, std::string& name, ModelConfig model_config,
            std::vector<int> device_list, Device::Ptr device);
  LmHead() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
  Module::Ptr col_parallel_linear;
  Module::Ptr all_gather;
};

}  // namespace llm_system