
#include "embedding.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"

namespace llm_system {

// EmbeddingLookup: leaf module that owns weight and produces activation //

EmbeddingLookup::EmbeddingLookup(std::string& prefix, std::string& name,
                                 ModelConfig model_config,
                                 std::vector<int> device_list,
                                 Device::Ptr device)
    // graph_execution=true: this is a leaf module that must run every iteration
    // to set the "Hidden vector" activation as ready
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  int parallel_num = device_list.size();

  std::vector<int> wgt_shape = {n_vocab / parallel_num, hidden_dimension};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr embedding = Tensor::Create("Embedding", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(embedding);
  add_tensor(output);
}

Tensor::Ptr EmbeddingLookup::forward(const Tensor::Ptr input,
                                     BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int n = model_config.hidden_dim;

  std::vector<int> shape = {m, n};
  // No compute time for embedding lookup (latency not modeled).
  // get_activation sets the tensor as ready for downstream AllReduce.
  Tensor::Ptr output = get_activation("Hidden vector", shape);
  return output;
}

// Embedding: wrapper module (graph_execution=false) like Decoder //

Embedding::Embedding(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    // Wrapper module (graph_execution=false) like Decoder — prevents duplicate
    // graph nodes from accumulating when forward() is re-invoked each iteration
    : Module(prefix, name, device, device_list),
      model_config(model_config) {

  // EmbeddingLookup: leaf that owns weight and produces hidden vector
  embedding_lookup = EmbeddingLookup::Create(module_map_name, "embedding_lookup",
                                              model_config, device_list, device);
  add_module(embedding_lookup);

  // AllReduce to exchange embeddings across TP devices after local lookup
  all_reduce = AllReduce::Create(module_map_name, "embedding_all_reduce",
                                  device_list, device);
  add_module(all_reduce);
}

Tensor::Ptr Embedding::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  // EmbeddingLookup: produces hidden vector (no compute, just activation ready)
  Tensor::Ptr output = (*embedding_lookup)(input, sequences_metadata);

  // AllReduce to gather embeddings from other TP devices that hold different vocab shards
  output = (*all_reduce)(output, sequences_metadata);

  return output;
}

}  // namespace llm_system
