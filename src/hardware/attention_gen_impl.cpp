#include <cstdint>
#include <limits>
#include <memory>

#include "common/assert.h"
#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "model/util.h"
#include "module/tensor.h"

namespace {
// Checked narrowing from int64_t to int for shape dimensions.
// Asserts that the value fits in int range; prevents silent wrap on overflow.
inline int checked_dim(int64_t v, const char* what) {
  assertTrue(v >= 0 && v <= std::numeric_limits<int>::max(),
             std::string("shape overflow at ") + what + ": " + std::to_string(v));
  return static_cast<int>(v);
}
} // namespace

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);
  // std::cout << " AttentionGenExecutionGPU called " << std::endl;
  // std::cout << " input shape: " << input->shape[0] << ", " << input->shape[1] << std::endl;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_read_bandwidth = config.memory_read_bandwidth;
  hw_metric memory_write_bandwidth = config.memory_write_bandwidth;
  hw_metric intermediate_bandwidth = config.intermediate_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;
  bool use_flash_attention = layer_info.use_flash_attention;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;
  time_ns total_memory_duration = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  ExecStatus temp;

  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  int num_seq = sequences_metadata->count_gen();

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if (use_flash_attention) {
    // Flash Attention: fused score + context
    time_ns total_interm_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim;
      n = seq->current_len + seq->num_process_token;
      // Chunked attention: non-anchor layers clamp KV length
      if (layer_info.use_chunked_attention)
        n = effective_kv_len(layer_info.decoder_idx, layer_info.chunked_attention_full_stride,
                             layer_info.chunked_attention_window, n);

      accumul_len += n;

      flops = 2.0 * m * k * n * num_heads + // score
              2.0 * m * k * n * num_heads; // context
      total_flops += flops;

      // HBM/HBF traffic: K cache + V cache reads
      double kv_read_size = (1.0 * n * k * num_kv_heads * 2) * input->precision_byte;
      // Intermediate traffic: Q read + output write
      double q_read   = (1.0 * m * k * num_heads) * input->precision_byte;
      double out_write = (1.0 * m * k * num_heads) * input->precision_byte;
      double interm_read  = q_read;
      double interm_write = out_write;
      total_memory_size += kv_read_size + interm_read + interm_write;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      time_ns read_duration   = kv_read_size / memory_read_bandwidth * 1e9;
      time_ns interm_duration = (interm_read + interm_write) / intermediate_bandwidth * 1e9;
      time_ns seq_mem_duration = computeMemoryDuration(read_duration, 0, interm_duration, config.intermediate_overlap);
      accumul_memory_duration += seq_mem_duration;
      total_memory_duration += seq_mem_duration;
      total_interm_duration += interm_duration;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AttentionGenGPU flash");

    if (use_ramulator) {
      // HBM/HBF path: issue only K cache and V cache reads
      // read key cache
      k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp += issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                             DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      // read value cache
      v_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp += issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                             DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);

      exec_status += temp;
      accumul_memory_duration = computeMemoryDuration(temp.memory_duration, 0,
                                                       total_interm_duration, config.intermediate_overlap);
    } else {
      // Ideal path: issue only K cache and V cache reads via ideal memory
      // Q read and output write are on intermediate buffer (already counted above)
      k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      v_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;
      // accumul_memory_duration is already correct from the per-seq loop above
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  } else {
    // Standard Attention: separate score, softmax, context

    // Scoring //
    time_ns total_interm_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim;
      n = seq->current_len + seq->num_process_token;
      // Chunked attention: non-anchor layers clamp KV length
      if (layer_info.use_chunked_attention)
        n = effective_kv_len(layer_info.decoder_idx, layer_info.chunked_attention_full_stride,
                             layer_info.chunked_attention_window, n);

      for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
        flops = m * k * n * 2.0 * attention_group_size;
        total_flops += flops;

        // K cache read
        double k_read = (1.0 * k * n) * input->precision_byte;
        // Intermediate traffic: Q read + score write
        double q_read      = (1.0 * m * k * num_heads / num_kv_heads) * input->precision_byte;
        double score_write = (1.0 * m * n * num_heads / num_kv_heads) * input->precision_byte;
        double interm_read  = q_read;
        double interm_write = score_write;
        total_memory_size += k_read + interm_read + interm_write;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        time_ns read_duration   = k_read / memory_read_bandwidth * 1e9;
        time_ns interm_duration = (interm_read + interm_write) / intermediate_bandwidth * 1e9;
        time_ns seq_mem_duration = computeMemoryDuration(read_duration, 0, interm_duration, config.intermediate_overlap);
        accumul_memory_duration += seq_mem_duration;
        total_memory_duration += seq_mem_duration;
        total_interm_duration += interm_duration;
      }
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AttentionGenGPU scoring");

    if (use_ramulator) {
      // issue only K cache reads
      k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp =
          issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    } else {
      // Ideal path: issue only K cache reads; Q and score are intermediate
      k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;
    }

    exec_status.total_duration +=
        std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;
      // Chunked attention: non-anchor layers clamp KV length
      if (layer_info.use_chunked_attention)
        n = effective_kv_len(layer_info.decoder_idx, layer_info.chunked_attention_full_stride,
                             layer_info.chunked_attention_window, n);

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;

      exec_status.total_duration += compute_duration;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    total_interm_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      // Chunked attention: non-anchor layers clamp KV length
      if (layer_info.use_chunked_attention)
        k = effective_kv_len(layer_info.decoder_idx, layer_info.chunked_attention_full_stride,
                             layer_info.chunked_attention_window, k);
      n = head_dim;

      for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
        flops = m * k * n * 2.0 * attention_group_size;
        total_flops += flops;

        // V cache read (one KV group)
        double v_read = (1.0 * k * n) * input->precision_byte;
        // Intermediate traffic: score read + output write
        double score_read = (1.0 * m * k * num_heads / num_kv_heads) * input->precision_byte;
        double out_write  = (1.0 * m * n * num_heads / num_kv_heads) * input->precision_byte;
        double interm_read  = score_read;
        double interm_write = out_write;
        total_memory_size += v_read + interm_read + interm_write;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        time_ns read_duration   = v_read / memory_read_bandwidth * 1e9;
        time_ns interm_duration = (interm_read + interm_write) / intermediate_bandwidth * 1e9;
        time_ns seq_mem_duration = computeMemoryDuration(read_duration, 0, interm_duration, config.intermediate_overlap);
        accumul_memory_duration += seq_mem_duration;
        total_memory_duration += seq_mem_duration;
        total_interm_duration += interm_duration;
      }
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "AttentionGenGPU context");

    if (use_ramulator) {
      // HBM/HBF path: issue only V cache reads
      v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
      temp =
          issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    } else {
      // Ideal path: issue only V cache reads; scores and output are intermediate
      v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;
    }

    exec_status.total_duration +=
        std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = total_memory_duration /
                            static_cast<double>(exec_status.total_duration);

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  int num_seq = sequences_metadata->count_gen();

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  });

  // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
  const int accumul_len_i = checked_dim(accumul_len, "AttentionGenLogic scoring");

  if (use_ramulator) {
    k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  });

  // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
  const int accumul_len_i2 = checked_dim(accumul_len, "AttentionGenLogic context");

  if (use_ramulator) {
    v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  int num_seq = sequences_metadata->count_gen();

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  });

  // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
  const int accumul_len_i = checked_dim(accumul_len, "AttentionGenPIM scoring");

  if (use_ramulator) {
    k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len_i, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  double opb = total_flops / total_memory_size;
  exec_status.total_duration += accumul_memory_duration * opb;

  // Softmax //

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  });

  // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
  const int accumul_len_i2 = checked_dim(accumul_len, "AttentionGenPIM context");

  if (use_ramulator) {
    v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else{
    v_cache->setShape({accumul_len_i2, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  opb = total_flops / total_memory_size;
  exec_status.total_duration += accumul_memory_duration * opb;

  // exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<int> orig_shape = input->shape;
  int num_seq = sequences_metadata->count_gen();

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenGPU flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenGPU flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read value
      input->setShape({accumul_len_i, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenGPU scoring");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenGPU scoring num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      input->setShape({num_heads, accumul_len_i});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "MLAGenGPU context");

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops =
    config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  int num_seq = sequences_metadata->count_gen();
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenLogic flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenLogic flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read value
      input->setShape({accumul_len_i, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenLogic scoring");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenLogic scoring num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      input->setShape({num_heads, accumul_len_i});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "MLAGenLogic context");

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  int num_seq = sequences_metadata->count_gen();
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int64_t accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenPIM flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenPIM flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read value
      input->setShape({accumul_len_i, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Score //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "MLAGenPIM scoring");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "MLAGenPIM scoring num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      input->setShape({num_heads, accumul_len_i});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len_i, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "MLAGenPIM context");

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len_i2, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  int num_seq = sequences_metadata->count_gen();

  std::vector<int> orig_shape = input->shape;

  int64_t accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenGPU flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "AbsorbMLAGenGPU flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Scoring for NoPE//
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenGPU NoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "AbsorbMLAGenGPU RoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i3 = checked_dim(accumul_len, "AbsorbMLAGenGPU context");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_kvr = checked_dim((int64_t)num_seq * kv_lora_rank, "AbsorbMLAGenGPU context num_seq*kv_lora_rank");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;


      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  // exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  int num_seq = sequences_metadata->count_gen();

  std::vector<int> orig_shape = input->shape;

  int64_t accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenLogic flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "AbsorbMLAGenLogic flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Scoring for NoPE//
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenLogic NoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "AbsorbMLAGenLogic RoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      // k_cache->setShape(shape);
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i3 = checked_dim(accumul_len, "AbsorbMLAGenLogic context");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_kvr = checked_dim((int64_t)num_seq * kv_lora_rank, "AbsorbMLAGenLogic context num_seq*kv_lora_rank");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  int num_seq = sequences_metadata->count_gen();

  std::vector<int> orig_shape = input->shape;

  int64_t accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  if(use_flash_mla){
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenPIM flash");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_nh = checked_dim((int64_t)num_seq * num_heads, "AbsorbMLAGenPIM flash num_seq*num_heads");

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({ns_nh, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len_i, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Scoring for NoPE//
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i = checked_dim(accumul_len, "AbsorbMLAGenPIM NoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i2 = checked_dim(accumul_len, "AbsorbMLAGenPIM RoPE scoring");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len_i2});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;

        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    });

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    sequences_metadata->for_each_gen([&](const Sequence::Ptr& seq) {

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;

      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    });

    // [Overflow hardening] Checked narrowing of accumulated sequence length for shape.
    const int accumul_len_i3 = checked_dim(accumul_len, "AbsorbMLAGenPIM context");
    // [Overflow hardening] Checked narrowing of dynamic shape dimension.
    const int ns_kvr = checked_dim((int64_t)num_seq * kv_lora_rank, "AbsorbMLAGenPIM context num_seq*kv_lora_rank");

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len_i3});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len_i3, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({ns_kvr, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

}  // namespace llm_system