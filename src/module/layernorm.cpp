#include "module/layernorm.h"
#include "scheduler/scheduler.h"
#include "common/assert.h"

namespace llm_system {

LayerNorm::LayerNorm(std::string& prefix, std::string& name,
                     int hidden_dim, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
    std::vector<int> shape = {hidden_dim, 1};

    Tensor::Ptr layer_norm_weight = Tensor::Create("layer_norm_weight", shape, "weight", device, device->model_config.precision_byte);
    Tensor::Ptr layer_norm_out = Tensor::Create("layer_norm_output", shape, "act", device, device->model_config.precision_byte);
    add_tensor(layer_norm_weight);
    add_tensor(layer_norm_out);
}

Tensor::Ptr LayerNorm::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];
  int n = 1;

  Tensor::Ptr layer_norm_weight = tensor_list.at("layer_norm_weight");
  Tensor::Ptr output = get_activation("layer_norm_output", input->shape);

  assertTrue(k == layer_norm_weight->shape[0], "input shape doesn't match with layer norm weight (= hidden dim)");

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_read_bandwidth = device->config.memory_read_bandwidth;
  hw_metric intermediate_bandwidth = device->config.intermediate_bandwidth;

  double flops, memory_size;
  flops =  4.0 * m * k; // layer norm (e.g. RMSNorm, Square (1) + Root (1) + Mean (2))
  memory_size = (2.0 * m * k + 1.0 * k * n) * input->precision_byte;

  // Read: input + weight, Write: output (with skip logic)
  double interm_read = (1.0 * m * k) * input->precision_byte;
  double interm_write = (1.0 * m * k) * input->precision_byte;
  double weight_read_size = (1.0 * k * n) * input->precision_byte;

  time_ns read_duration   = weight_read_size / memory_read_bandwidth * 1e9;
  time_ns interm_duration = (interm_read + interm_write) / intermediate_bandwidth * 1e9;
  time_ns memory_duration = computeMemoryDuration(read_duration, 0, interm_duration, device->config.intermediate_overlap);
  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  

  time_ns total_time = std::max(compute_duration, memory_duration);

  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

}  // namespace llm_system