
#include "module/attention.h"

#include <cstdint>
#include <limits>

#include "common/assert.h"
#include "hardware/hardware_config.h"
#include "model/util.h"
#include "module/base.h"

namespace {
// Checked narrowing from int64_t to int for shape dimensions.
inline int checked_dim(int64_t v, const char* what) {
  assertTrue(v >= 0 && v <= std::numeric_limits<int>::max(),
             std::string("shape overflow at ") + what + ": " + std::to_string(v));
  return static_cast<int>(v);
}
} // namespace

namespace llm_system {

// SelfAttentionGen //

SelfAttentionGen::SelfAttentionGen(std::string& prefix, std::string& name,
                                   int head_dim, int num_heads,
                                   int num_kv_heads, int max_seq_len,
                                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                                   Device::Ptr device, int decoder_idx)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim),
      decoder_idx(decoder_idx) {
  int parallel_num = device_list.size();

  // Chunked attention: non-anchor layers allocate smaller KV cache
  int effective_max_seq_len = effective_kv_len(device->model_config, decoder_idx, max_seq_len);
  std::vector<int> shape = {effective_max_seq_len, head_dim};

  // [Memory optimization] Instead of creating batch_size * num_kv_heads * 2 cache tensors,
  // keep only one representative K and V tensor for timing simulation (setShape-based).
  // The full pre-allocated cache footprint is tracked analytically via logical_cache_bytes_.
  
  Tensor::Ptr k_cache = Tensor::Create(
      "k_cache_0_0", shape, "cache", device, device->model_config.precision_byte);
  k_cache->setCountInFootprint(false);
  add_tensor(k_cache);

  Tensor::Ptr v_cache = Tensor::Create(
      "v_cache_0_0", shape, "cache", device, device->model_config.precision_byte);
  v_cache->setCountInFootprint(false);
  add_tensor(v_cache);

  // Equivalent to: batch_size * num_kv_heads * 2 tensors, each of size
  // effective_max_seq_len * head_dim * precision_byte
  long long logical_bytes = (long long)batch_size * num_kv_heads * 2
                          * effective_max_seq_len * head_dim
                          * device->model_config.precision_byte;
  addLogicalCacheBytes(logical_bytes);

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr SelfAttentionGen::forward(const Tensor::Ptr input,
                                      BatchedSequence::Ptr sequences_metadata) {
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.low_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformLow();
  }

  layer_info.attention_group_size = num_heads / num_kv_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  // Chunked attention: pass layer index and config to hardware execution
  layer_info.decoder_idx = this->decoder_idx;
  layer_info.use_chunked_attention = device->model_config.use_chunked_attention;
  layer_info.chunked_attention_window = device->model_config.chunked_attention_window;
  layer_info.chunked_attention_full_stride = device->model_config.chunked_attention_full_stride;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  tensor_list.push_back(get_cache("k_cache", 0, 0, false));
  tensor_list.push_back(get_cache("v_cache", 0, 0, false));

  device->execution(LayerType::ATTENTION_GEN, tensor_list, sequences_metadata,
                    layer_info);

  return output_tensor;
}

// SelfAttentionSum //
SelfAttentionSum::SelfAttentionSum(std::string& prefix, std::string& name,
                                   int head_dim, int num_heads,
                                   int num_kv_heads, int max_seq_len,
                                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                                   Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr SelfAttentionSum::forward(const Tensor::Ptr input,
                                      BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::ATTENTION_SUM, tensor_list, sequences_metadata,
                    layer_info);

  return output_tensor;
}

// SelfAttentionMixed //
SelfAttentionMixed::SelfAttentionMixed(std::string& prefix, std::string& name,
                                       int head_dim, int num_heads,
                                       int num_kv_heads, int max_seq_len,
                                       int batch_size, int qk_rope_head_dim,
                                       std::vector<int> device_list,
                                       Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr SelfAttentionMixed::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  tensor_list.push_back(input);

  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  device->execution(LayerType::ATTENTION_MIXED, tensor_list, sequences_metadata,
                    layer_info);

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  return output_tensor;
}

// AttentionSplit //
AttentionSplit::AttentionSplit(std::string& prefix, std::string& name,
                               int head_dim, int num_heads, int num_kv_heads,
                               int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                               bool use_absorb, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim),
      kv_lora_rank(kv_lora_rank),
      use_absorb(use_absorb) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {1, 1};

  Tensor::Ptr sum_tensor = Tensor::Create("sum_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(sum_tensor);
  Tensor::Ptr gen_tensor = Tensor::Create("gen_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(gen_tensor);
}

TensorVec AttentionSplit::forward(const TensorVec input,
                                  BatchedSequence::Ptr sequences_metadata) {
  // [Overflow hardening] Checked narrowing of token counts used as shape dimensions.
  int sum_token = checked_dim(sequences_metadata->get_sum_process_token(),
                              "AttentionSplit sum_token");
  int gen_token = checked_dim(sequences_metadata->get_gen_process_token(),
                              "AttentionSplit gen_token");

  TensorVec tensor_list;
  std::vector<int> sum_shape = {1, 1};
  std::vector<int> gen_shape = {1, 1};

  if(qk_rope_head_dim == 0){
    sum_shape = {sum_token, (num_heads + num_kv_heads * 2) * head_dim};
    gen_shape = {gen_token, (num_heads + num_kv_heads * 2) * head_dim};
  }
  else{
    if(use_absorb){
      int num_heads_per_device = input.size();
      sum_shape = {num_heads_per_device, sum_token, kv_lora_rank};
      gen_shape = {num_heads_per_device, gen_token, kv_lora_rank};
    }
    else{
      sum_shape = {sum_token, num_heads * (2 * (head_dim + qk_rope_head_dim) + head_dim )};
      gen_shape = {gen_token, num_heads * (2 * (head_dim + qk_rope_head_dim) + head_dim )};
    }
  }
  Tensor::Ptr sum_tensor = get_activation("sum_tensor", sum_shape);
  Tensor::Ptr gen_tensor = get_activation("gen_tensor", gen_shape);

  device->status.device_time =
      std::max(device->status.device_time,
              std::max(device->status.low_time, device->status.high_time));
  device->status.high_time = device->status.device_time;
  device->status.low_time = device->status.device_time;

  if (device->config.parallel_execution) {
    sum_tensor->setPerformHigh();
    gen_tensor->setPerformLow();
  }

  tensor_list.resize(0);
  if(use_absorb){
    tensor_list.push_back(sum_tensor);
    tensor_list.push_back(gen_tensor);
  }
  else{
    tensor_list.push_back(sum_tensor);
    tensor_list.push_back(gen_tensor);
  }
  return tensor_list;
}

// AttentionMerge //
AttentionMerge::AttentionMerge(std::string& prefix, std::string& name,
                               int head_dim, int num_heads, int num_kv_heads,
                               int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                               bool use_absorb, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      kv_lora_rank(kv_lora_rank),
      use_absorb(use_absorb) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {1, 1};

  Tensor::Ptr output_tensor =
      Tensor::Create("output_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output_tensor);
}

TensorVec AttentionMerge::forward(const TensorVec input,
                                  BatchedSequence::Ptr sequences_metadata) {
  // [Overflow hardening] Keep as int64_t; check the final merged dimension at shape use.
  // Precondition: AttentionSplit::forward() already checked sum_token/gen_token individually.
  int64_t sum_token = sequences_metadata->get_sum_process_token();
  int64_t gen_token = sequences_metadata->get_gen_process_token();

  if(use_absorb){

    int merged_token = checked_dim(sum_token + gen_token, "AttentionMerge merged_token");
    std::vector<int> shape = {merged_token, kv_lora_rank};
    Tensor::Ptr output_tensor = get_activation("output_tensor", shape);

    device->status.device_time =
        std::max(device->status.device_time,
                std::max(device->status.low_time, device->status.high_time));
    device->status.high_time = device->status.device_time;
    device->status.low_time = device->status.device_time;

    TensorVec tensor_list;
    tensor_list.resize(0);
    for(int i = 0 ; i < num_heads; i ++){
      tensor_list.push_back(output_tensor);
    }

    return tensor_list;
  }
  else{
    // Narrow for assert comparison (dim(0) returns int).
    int sum_token_i = checked_dim(sum_token, "AttentionMerge sum_token");
    int gen_token_i = checked_dim(gen_token, "AttentionMerge gen_token");
    assertTrue(input[0]->dim(0) == sum_token_i,
              "Dimenison of input tensor of AttentionMerge is not matched");
    assertTrue(input[1]->dim(0) == gen_token_i,
              "Dimenison of input tensor of AttentionMerge is not matched");

    int merged_token = checked_dim(sum_token + gen_token, "AttentionMerge merged_token");
    std::vector<int> shape = {merged_token, num_heads * head_dim};
    Tensor::Ptr output_tensor = get_activation("output_tensor", shape);

    device->status.device_time =
        std::max(device->status.device_time,
                std::max(device->status.low_time, device->status.high_time));
    device->status.high_time = device->status.device_time;
    device->status.low_time = device->status.device_time;

    TensorVec tensor_list;
    tensor_list.resize(0);
    tensor_list.push_back(output_tensor);

    return tensor_list;
  }
}

// MultiLatentAttentionGen //

MultiLatentAttentionGen::MultiLatentAttentionGen(std::string& prefix, std::string& name,
    int head_dim, int num_heads,
    int num_kv_heads, int max_seq_len,
    int batch_size, int qk_rope_head_dim, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv),
  use_flash_mla(use_flash_mla) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {max_seq_len, head_dim};
  if (!compressed_kv) {
    // [Memory optimization] Keep only representative K/V tensors for timing.
    Tensor::Ptr k_cache = Tensor::Create(
        "k_cache_0_0", shape, "cache", device, device->model_config.precision_byte);
    k_cache->setCountInFootprint(false);
    add_tensor(k_cache);

    Tensor::Ptr v_cache = Tensor::Create(
        "v_cache_0_0", shape, "cache", device, device->model_config.precision_byte);
    v_cache->setCountInFootprint(false);
    add_tensor(v_cache);

    // Equivalent to: batch_size * num_kv_heads * 2 tensors, each of size
    // max_seq_len * head_dim * precision_byte
    long long logical_bytes = (long long)batch_size * num_kv_heads * 2
                            * max_seq_len * head_dim
                            * device->model_config.precision_byte;
    addLogicalCacheBytes(logical_bytes);
  }

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr MultiLatentAttentionGen::forward(const Tensor::Ptr input,
      BatchedSequence::Ptr sequences_metadata) {
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.low_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformLow();
  }else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.num_heads = num_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.use_flash_mla = use_flash_mla;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  if(compressed_kv){
    
  }
  else{
    tensor_list.push_back(get_cache("k_cache", 0, 0, false));
    tensor_list.push_back(get_cache("v_cache", 0, 0, false));
  }

  device->execution(LayerType::MLA_GEN, tensor_list, sequences_metadata, layer_info);

  return output_tensor;
}

// MultiLatentAttentionSum //
MultiLatentAttentionSum::MultiLatentAttentionSum(std::string& prefix, std::string& name,
  int head_dim, int num_heads,
  int num_kv_heads, int max_seq_len,
  int batch_size, int qk_rope_head_dim, bool use_flash_attention, std::vector<int> device_list,
  Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  use_flash_attention(use_flash_attention) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr MultiLatentAttentionSum::forward(const Tensor::Ptr input,
     BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = input->shape;
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.parallel_execution = true;
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }
  else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.use_flash_attention = use_flash_attention;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::MLA_SUM, tensor_list, sequences_metadata,
  layer_info);

  return output_tensor;
}

// MultiLatentAttentionGen //

AbsorbMLAGen::AbsorbMLAGen(std::string& prefix, std::string& name,
    int head_dim, int num_heads,
    int num_kv_heads, int max_seq_len,
    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool compressed_kv, 
    bool use_flash_mla, std::vector<int> device_list, Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  kv_lora_rank(kv_lora_rank),
  compressed_kv(compressed_kv),
  use_flash_mla(use_flash_mla) {
  int parallel_num = device_list.size();

  std::vector<int> latent_kv_shape = {max_seq_len, kv_lora_rank};
  std::vector<int> latent_pe_shape = {max_seq_len, qk_rope_head_dim};
  std::vector<int> out_shape = {num_kv_heads / parallel_num, max_seq_len, kv_lora_rank};

  // [Memory optimization] Keep only representative latent cache tensors for timing.
  Tensor::Ptr latent_kv_cache = Tensor::Create(
      "latent_kv_cache_0", latent_kv_shape, "cache", device, device->model_config.precision_byte);
  latent_kv_cache->setCountInFootprint(false);
  add_tensor(latent_kv_cache);

  Tensor::Ptr latent_pe_cache = Tensor::Create(
      "latent_pe_cache_0", latent_pe_shape, "cache", device, device->model_config.precision_byte);
  latent_pe_cache->setCountInFootprint(false);
  add_tensor(latent_pe_cache);

  // Equivalent to: batch_size tensors of shape {max_seq_len, kv_lora_rank}
  // + batch_size tensors of shape {max_seq_len, qk_rope_head_dim}
  long long logical_bytes = (long long)batch_size
                          * ((long long)max_seq_len * kv_lora_rank
                             + (long long)max_seq_len * qk_rope_head_dim)
                          * device->model_config.precision_byte;
  addLogicalCacheBytes(logical_bytes);

  Tensor::Ptr output = Tensor::Create("attn_output", out_shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AbsorbMLAGen::forward(const Tensor::Ptr input,
      BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output_tensor = get_activation("attn_output", input->shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  } else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.num_heads = num_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.kv_lora_rank = kv_lora_rank;
  layer_info.use_flash_mla = use_flash_mla;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  // [Memory optimization] Original code looped over all gen sequences, but the
  // returned cache pointers were never passed to execution (AbsorbMLAGenExecutionGPU
  // only uses tensor.at(0) = input). Replaced with single representative lookup.
  if (sequences_metadata->has_gen()) {
    get_cache("latent_kv_cache", 0, 0, true);
    get_cache("latent_pe_cache", 0, 0, true);
  }

  device->execution(LayerType::ABSORBED_MLA_GEN, tensor_list, sequences_metadata, layer_info);
  
  return output_tensor;
}

// MultiLatentAttentionSum //
AbsorbMLASum::AbsorbMLASum(std::string& prefix, std::string& name,
  int head_dim, int num_heads,
  int num_kv_heads, int max_seq_len,
  int batch_size, int qk_rope_head_dim, int kv_lora_rank, std::vector<int> device_list,
  Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  kv_lora_rank(kv_lora_rank) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {num_kv_heads / parallel_num, max_seq_len, kv_lora_rank};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AbsorbMLASum::forward(const Tensor::Ptr input,
     BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = input->shape;
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  } else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.kv_lora_rank = kv_lora_rank;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::ABSORBED_MLA_SUM, tensor_list, sequences_metadata,
  layer_info);

  
  return output_tensor;
}

}  // namespace llm_system