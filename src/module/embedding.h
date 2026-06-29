#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "module/communication.h"  // AllReduce
#include "scheduler/sequence.h"

namespace llm_system {

// EmbeddingLookup: leaf module (graph_execution=true) that owns the weight
// and produces the hidden vector activation. No compute time (lookup latency
// not modeled), but the activation tensor becomes ready for downstream use.
class EmbeddingLookup : public Module {
 public:
  using Ptr = std::shared_ptr<EmbeddingLookup>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new EmbeddingLookup(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  EmbeddingLookup(std::string& prefix, std::string& name, ModelConfig model_config,
                  std::vector<int> device_list, Device::Ptr device);
  EmbeddingLookup() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

// Embedding: wrapper module (graph_execution=false) like Decoder.
// Composes EmbeddingLookup -> AllReduce.
// This separation avoids duplicate graph nodes that would accumulate
// if a graph_execution=true module called operator() on sub-modules.
class Embedding : public Module {
 public:
  using Ptr = std::shared_ptr<Embedding>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new Embedding(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Embedding(std::string& prefix, std::string& name, ModelConfig model_config,
            std::vector<int> device_list, Device::Ptr device);
  Embedding() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
  Module::Ptr embedding_lookup;
  Module::Ptr all_reduce;
};

}  // namespace llm_system
