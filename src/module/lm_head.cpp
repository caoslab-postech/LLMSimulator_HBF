
#include "lm_head.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// LmHead //

namespace llm_system {

LmHead::LmHead(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    // Wrapper module (graph_execution=false) like Decoder — prevents duplicate
    // graph nodes from accumulating when forward() is re-invoked each iteration
    : Module(prefix, name, device, device_list),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;

  // Column-parallel LmHead: ColumnParallelLinear shards n_vocab across TP devices
  col_parallel_linear = ColumnParallelLinear::Create(
      module_map_name, "lm_head_col_linear",
      hidden_dimension, n_vocab, device_list, device);
  add_module(col_parallel_linear);

  // AllGather to collect logits from all TP devices for output token selection
  all_gather = AllGather::Create(module_map_name, "lm_head_all_gather",
                                  device_list, device);
  add_module(all_gather);
}

Tensor::Ptr LmHead::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  // Column-parallel GEMM: [batch, hidden_dim] x [hidden_dim, n_vocab/TP]
  Tensor::Ptr linear_out = (*col_parallel_linear)(input, sequences_metadata);
  // AllGather: collect partial logits from all TP devices
  Tensor::Ptr output = (*all_gather)(linear_out, sequences_metadata);
  return output;
}

}  // namespace llm_system