#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_read_bandwidth = config.memory_read_bandwidth;
  hw_metric intermediate_bandwidth = config.intermediate_bandwidth;

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;

  // weight read
  hw_metric weight_read_size = (k * n) * weight->precision_byte;
  // input read + output write
  hw_metric interm_read_size  = (m * k) * weight->precision_byte;
  hw_metric interm_write_size = (m * n) * weight->precision_byte;

  hw_metric total_memory_size = weight_read_size + interm_read_size + interm_write_size;

  time_ns compute_duration = total_flops / compute_peak_flops * 1e9;
  time_ns read_duration    = weight_read_size / memory_read_bandwidth * 1e9;
  time_ns interm_duration  = (interm_read_size + interm_write_size) / intermediate_bandwidth * 1e9;
  time_ns memory_duration  = computeMemoryDuration(read_duration, 0, interm_duration,
                                                   config.intermediate_overlap);

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1e9 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util =
      static_cast<double>(memory_duration) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

// QKV fused linear for GPU: splits output write into Q (intermediate_bw) and KV (memory_write_bw)
ExecStatus QKVLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                                  Tensor_Ptr weight, Tensor_Ptr output,
                                  bool use_ramulator, const LayerInfo& layer_info) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_read_bandwidth = config.memory_read_bandwidth;
  hw_metric memory_write_bandwidth = config.memory_write_bandwidth;
  hw_metric intermediate_bandwidth = config.intermediate_bandwidth;

  double m = input->shape[0];   // batch * seq_len
  double k = input->shape[1];   // hidden_dim
  double n = weight->shape[1];  // head_dim * (num_heads + 2*num_kv_heads) / TP

  // // DEBUG: verify QKVLinearExecutionGPU is called with correct params
  // std::cout << "[QKVLinearExecutionGPU] m=" << m << " k=" << k << " n=" << n
  //           << " num_heads=" << layer_info.qkv_num_heads
  //           << " num_kv_heads=" << layer_info.qkv_num_kv_heads
  //           << " head_dim=" << layer_info.qkv_head_dim
  //           << " read_bw=" << memory_read_bandwidth
  //           << " write_bw=" << memory_write_bandwidth
  //           << " interm_bw=" << intermediate_bandwidth << std::endl;

  // TP-sharded head counts from LayerInfo
  int num_heads = layer_info.qkv_num_heads;
  int num_kv_heads = layer_info.qkv_num_kv_heads;
  int head_dim = layer_info.qkv_head_dim;

  int q_dim = head_dim * num_heads;             // Q output dimension
  int kv_dim = head_dim * 2 * num_kv_heads;     // KV output dimension

  hw_metric total_flops = 2.0 * m * k * n;

  // Weight read from HBM
  hw_metric weight_read_size = k * n * weight->precision_byte;
  // Input read from intermediate buffer
  hw_metric interm_read_size = m * k * weight->precision_byte;
  // Q output write
  hw_metric q_write_size = m * q_dim * weight->precision_byte;
  // KV output write to HBM/HBF (KV cache)
  hw_metric kv_write_size = m * kv_dim * weight->precision_byte;

  // Accumulate KV write for PEC tracking (HBF only)
  if (device->config.gpu_gen.rfind("HBF", 0) == 0) {
    device->cumulative_kv_write_bytes += (long long)kv_write_size;
  }

  hw_metric total_memory_size = weight_read_size + interm_read_size
                                + q_write_size + kv_write_size;

  time_ns compute_duration = total_flops / compute_peak_flops * 1e9;
  // HBM read: weight
  time_ns read_duration    = weight_read_size / memory_read_bandwidth * 1e9;
  // HBM write: KV output to KV cache
  time_ns write_duration   = kv_write_size / memory_write_bandwidth * 1e9;
  // Intermediate: input read + Q output write
  time_ns interm_duration  = (interm_read_size + q_write_size) / intermediate_bandwidth * 1e9;
  time_ns memory_duration  = computeMemoryDuration(read_duration, write_duration,
                                                    interm_duration,
                                                    config.intermediate_overlap);

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU,
                                         DRAMRequestType::kRead, input);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU,
                                         DRAMRequestType::kRead, weight);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU,
                                         DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1e9 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util =
      static_cast<double>(memory_duration) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  assertTrue(input->shape.size() == 3, "input tensor is not 3D tensor");
  assertTrue(weight->shape.size() == 3, "weight tensor is not 3D tensor");
  assertTrue(output->shape.size() == 3, "output tensor is not 3D tensor");

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_read_bandwidth = config.memory_read_bandwidth;
  hw_metric intermediate_bandwidth = config.intermediate_bandwidth;

  int num_heads = input->shape[0];
  std::vector<int> input_orig_shape  = input->shape;
  std::vector<int> weight_orig_shape = weight->shape;
  std::vector<int> output_orig_shape = output->shape;

  int m = input->shape[1];
  int k = input->shape[2];
  int n = weight->shape[2];

  hw_metric total_flops = 2.0 * m * k * n * 1.0 * num_heads;

  // HBM/HBF path: weight read only
  hw_metric weight_read_size;
  // Intermediate path: input read + output write
  hw_metric interm_read_size;
  hw_metric interm_write_size;

  if (duplicated_input) {
    weight_read_size  = 1.0 * k * n * num_heads * weight->precision_byte;
    interm_read_size  = 1.0 * m * k * weight->precision_byte;
    interm_write_size = 1.0 * m * n * num_heads * weight->precision_byte;
  } else {
    weight_read_size  = 1.0 * k * n * num_heads * weight->precision_byte;
    interm_read_size  = 1.0 * m * k * num_heads * weight->precision_byte;
    interm_write_size = 1.0 * m * n * num_heads * weight->precision_byte;
  }

  hw_metric total_memory_size = weight_read_size + interm_read_size + interm_write_size;

  time_ns compute_duration = total_flops / compute_peak_flops * 1e9;
  time_ns read_duration    = weight_read_size / memory_read_bandwidth * 1e9;
  time_ns interm_duration  = (interm_read_size + interm_write_size) / intermediate_bandwidth * 1e9;
  time_ns memory_duration  = computeMemoryDuration(read_duration, 0, interm_duration,
                                                   config.intermediate_overlap);

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    // Only issue weight to HBM/HBF ramulator; input/output are on intermediate buffer
    if (duplicated_input) {
      input->setShape({m, k});
    } else {
      input->setShape({m, k * num_heads});
    }
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    
    weight->setShape({k * num_heads, n});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);

    output->setShape({m, n * num_heads});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1e9 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util =
      static_cast<double>(memory_duration) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  exec_status.opb = total_flops / total_memory_size;

  input->setShape(input_orig_shape);
  weight->setShape(weight_orig_shape);
  output->setShape(output_orig_shape);

  return exec_status;
};

ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

}  // namespace llm_system