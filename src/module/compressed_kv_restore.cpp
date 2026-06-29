#include "module/compressed_kv_restore.h"

#include <cstdint>
#include <limits>

#include "common/assert.h"
#include "hardware/hardware_config.h"
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

CompressedKVRestore::CompressedKVRestore(std::string& prefix, std::string& name,
        int head_dim, int num_heads, 
        int max_seq_len, int batch_size, int kv_lora_rank, int qk_rope_head_dim,
        bool compressed_kv, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
    head_dim(head_dim),
    kv_lora_rank(kv_lora_rank),
    qk_rope_head_dim(qk_rope_head_dim),
    compressed_kv(compressed_kv) {
    int parallel_num = device_list.size();

    std::vector<int> c_kv_shape = {max_seq_len, kv_lora_rank};
    std::vector<int> c_pe_shape = {max_seq_len, qk_rope_head_dim};
    std::vector<int> output_shape = {1,1};
    if (compressed_kv) {
        // [Memory optimization] Keep only representative compressed cache tensors.
        Tensor::Ptr c_kv_cache = Tensor::Create(
            "c_kv_cache_0", c_kv_shape, "cache", device, device->model_config.precision_byte);
        c_kv_cache->setCountInFootprint(false);
        add_tensor(c_kv_cache);

        Tensor::Ptr c_pe_cache = Tensor::Create(
            "c_pe_cache_0", c_pe_shape, "cache", device, device->model_config.precision_byte);
        c_pe_cache->setCountInFootprint(false);
        add_tensor(c_pe_cache);

        // Equivalent to: batch_size tensors of shape {max_seq_len, kv_lora_rank}
        // + batch_size tensors of shape {max_seq_len, qk_rope_head_dim}
        long long logical_bytes = (long long)batch_size
                                * ((long long)max_seq_len * kv_lora_rank
                                   + (long long)max_seq_len * qk_rope_head_dim)
                                * device->model_config.precision_byte;
        addLogicalCacheBytes(logical_bytes);
    }
    Tensor::Ptr concated_kv_out = Tensor::Create("concated_latent_kv", output_shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr concated_k_rope_out = Tensor::Create("concated_k_rope", output_shape, "act", device, device->model_config.precision_byte);
    add_tensor(concated_kv_out);
    add_tensor(concated_k_rope_out);
}

TensorVec CompressedKVRestore::forward(const TensorVec input,
        BatchedSequence::Ptr sequences_metadata) {
    int64_t sum_token = sequences_metadata->get_sum_process_token();
    int64_t gen_total_token = sequences_metadata->get_total_sequence_length(); // sum of current len of gen sequences

    TensorVec tensor_list;
    std::vector<int>latent_kv_shape;
    std::vector<int>k_rope_shape;

    if(compressed_kv){
        int num_gen = sequences_metadata->count_gen();
        // [Overflow hardening] Checked narrowing of total token dimension for shape
        int total_tokens_i = checked_dim(sum_token + gen_total_token + num_gen,
                                         "CompressedKVRestore total_tokens");
        latent_kv_shape = {total_tokens_i, kv_lora_rank};
        k_rope_shape = {total_tokens_i, qk_rope_head_dim};
        Tensor::Ptr c_kv_cache = get_cache("c_kv_cache", 0, 0, true);
        Tensor::Ptr c_pe_cache = get_cache("c_pe_cache", 0, 0, true);
    }
    else{
        latent_kv_shape = input[0]->shape;
        k_rope_shape = input[1]->shape;
    }
    
    Tensor::Ptr concated_kv_out = get_activation("concated_latent_kv", latent_kv_shape);
    Tensor::Ptr concated_k_rope_out = get_activation("concated_k_rope", k_rope_shape);

    tensor_list.resize(0);
    tensor_list.push_back(concated_kv_out);
    tensor_list.push_back(concated_k_rope_out);

    return tensor_list;
}

} // namespace llm_system