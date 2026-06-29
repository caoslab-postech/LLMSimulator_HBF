#include "module/communication.h"
#include "hardware/cluster.h"
#include "model/util.h"
#include "scheduler/scheduler.h"

#include "common/assert.h"
// AllReduce //

namespace llm_system {

AllReduce::AllReduce(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) { // sync == true, thererfore need to be synced
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("allreduce_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AllReduce::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  Tensor::Ptr output = get_activation("allreduce_output", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  int hop = (device_list.size() - 1) * 2;
  size /= device_list.size();

  time_ns one_hop =
      device->config.device_ict_latency +
      size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = one_hop * hop;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

AllGather::AllGather(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("allgather_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AllGather::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {

  // Output: gather dimension expanded by device count
  std::vector<int> out_shape = {input->shape[0], input->shape[1] * (int)device_list.size()};
  Tensor::Ptr output = get_activation("allgather_output", out_shape);

  long size = input->getSize();  // local shard size (already 1/N of full)
  if (size == 0 || device_list.size() <= 1) {
    return output;
  }

  // Ring AllGather: (N-1) hops, each hop transfers full local shard
  int hop = device_list.size() - 1;

  time_ns one_hop =
      device->config.device_ict_latency +
      size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = one_hop * hop;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

AllScatter::AllScatter(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  parallel_num = device_list.size();

  Tensor::Ptr output = Tensor::Create("allscatter_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AllScatter::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  long m = input->shape[0];
  long k = input->shape[1];

  std::vector<int> shape = {input->shape[0], input->shape[1] / parallel_num};

  Tensor::Ptr output = get_activation("allscatter_output", shape);

  return output;
}

MoEScatter::MoEScatter(std::string& prefix, std::string& name,
                       std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_scatter_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr MoEScatter::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_scatter_output", input->shape);

  // With PP, MoE device_list = stage group (all DP replicas in same PP stage),
  // so cross-DP communication must occur.

  if (!device->perform_execution) {
    return output;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Send time //
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int src = device->device_total_rank; // current device
  int src_node = src / device->config.num_device;

  int ne_tp_dg = device->model_config.ne_tp_dg;
  int e_tp_dg = device->model_config.e_tp_dg;

  // TP sharing devices within same TP group (always contiguous)
  std::vector<int> tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;
  for(int device_idx = device_list_offset; device_idx < device_list_offset + ne_tp_dg; device_idx ++){
    tp_sharing_device_list.push_back(device_idx);
  }
  std::unordered_set<int> set_tp_devices(tp_sharing_device_list.begin(), tp_sharing_device_list.end());

  // Stage-scoped: only devices in this PP stage's MoE group
  int total_num_device = device_list.size();

  // Iterate over stage device list (possibly non-contiguous)
  for(int dst : device_list){ // dst: destination device (global rank)
    if(set_tp_devices.count(dst) == 0){ // outer tp space
      // Use stage-local rank for expert offset calculation
      int dst_local = get_local_rank_in_list(device_list, dst);
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (dst_local / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

      int dst_node_id = dst / device->config.num_device;
      if(dst_node_id == src_node){
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
      else{
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg
  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;
  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;
    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Receive time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int dst_device_rank = device->device_total_rank;
  int dst_node = dst_device_rank / device->config.num_device;

  // Stage-local TP group index within the stage device list
  int my_local_rank = get_local_rank_in_list(device_list, device->device_total_rank);
  int dst_dp_rank = my_local_rank / ne_tp_dg;

  int expert_id_offset_recv = device->model_config.num_routed_expert / total_num_device * (my_local_rank / e_tp_dg) * e_tp_dg;
  int num_expert_per_device_recv = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  int num_tp_groups = total_num_device / ne_tp_dg;
  for(int src_dp_idx = 0; src_dp_idx < num_tp_groups; src_dp_idx ++){
    if(src_dp_idx != dst_dp_rank){

      // Map stage TP group index to actual global rank via device_list
      int src_device_rank = device_list[src_dp_idx * ne_tp_dg];
      int src_node = src_device_rank / device->config.num_device;

      if(dst_node == src_node){
        // intra node
        for(int e_id = expert_id_offset_recv; e_id < expert_id_offset_recv + num_expert_per_device_recv; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
      else{
        // inter node
        for(int e_id = expert_id_offset_recv; e_id < expert_id_offset_recv + num_expert_per_device_recv; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand
  
    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;
  
    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
      // device->status.device_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

MoEGather::MoEGather(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_gather_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

TensorVec MoEGather::forward(const TensorVec input_vec,
                             BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr input = input_vec.at(0);
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_gather_output", input->shape);

  // With PP, MoE device_list = stage group across DP replicas.

  if (!device->perform_execution) {
    return input_vec;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Receive time //
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int dst = device->device_total_rank;
  int dst_node = dst / device->config.num_device;

  int e_tp_dg = device->model_config.e_tp_dg;
  int ne_tp_dg = device->model_config.ne_tp_dg;

  // e_tp sharing devices (always contiguous within same DP replica)
  std::vector<int> e_tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / e_tp_dg * e_tp_dg;
  for(int device_idx = device_list_offset; device_idx < device_list_offset + e_tp_dg; device_idx ++){
    e_tp_sharing_device_list.push_back(device_idx);
  }
  std::unordered_set<int> set_e_tp_devices(e_tp_sharing_device_list.begin(), e_tp_sharing_device_list.end());

  int total_num_device = device_list.size();

  for(int src : device_list){
    if(set_e_tp_devices.count(src) == 0){
      int src_local = get_local_rank_in_list(device_list, src);
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (src_local / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

      int src_node_id = src / device->config.num_device;
      if(src_node_id == dst_node){
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
      else{
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= device->model_config.e_tp_dg;
  inter_node_comm_size /= device->model_config.e_tp_dg;

  intra_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce
  inter_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce

  // FP8 dispatch && BF16 combine
  if((device->model_config.model_name == "deepseekV3") && device->model_config.precision_byte == 1){
    intra_node_comm_size *= 2;
    inter_node_comm_size *= 2;
  }

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand
  
    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;
  
    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Send time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int src_device_rank = device->device_total_rank;
  int src_node = src_device_rank / device->config.num_device;

  int my_local_rank_g = get_local_rank_in_list(device_list, device->device_total_rank);
  int src_dp_rank = my_local_rank_g / ne_tp_dg;

  int expert_id_offset_send = device->model_config.num_routed_expert / total_num_device * (my_local_rank_g / e_tp_dg) * e_tp_dg;
  int num_expert_per_device_send = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  int num_tp_groups_g = total_num_device / ne_tp_dg;
  for(int dst_dp_idx = 0; dst_dp_idx < num_tp_groups_g; dst_dp_idx ++){
    if(dst_dp_idx != src_dp_rank){

      int dst_device_rank = device_list[dst_dp_idx * ne_tp_dg];
      int dst_node_g = dst_device_rank / device->config.num_device;

      if(dst_node_g == src_node){
        for(int e_id = expert_id_offset_send; e_id < expert_id_offset_send + num_expert_per_device_send; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
      else{
        for(int e_id = expert_id_offset_send; e_id < expert_id_offset_send + num_expert_per_device_send; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand
  
    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;
  
    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return input_vec;
}

Sync::Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
           Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  std::vector<int> shape = {1, 1};

  auto sync__set =
      Sync__Set::Create(module_map_name, "sync__set", device_list, device);
  add_module(sync__set);

  auto sync__ = Sync__::Create(module_map_name, "sync__", device_list, device);
  add_module(sync__);
}

Tensor::Ptr Sync::forward(const Tensor::Ptr input,
                          BatchedSequence::Ptr sequences_metadata) {
  auto sync__set = get_module("sync__set");
  auto sync__ = get_module("sync__");

  Tensor::Ptr temp = (*sync__set)(input, sequences_metadata);
  Tensor::Ptr output = (*sync__)(temp, sequences_metadata);
  return output;
}

Sync__Set::Sync__Set(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("sync", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr Sync__Set::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output = get_activation("sync");
  device->status.device_time =
      std::max(device->status.device_time,
               std::max(device->status.low_time, device->status.high_time));
  device->status.high_time = device->status.device_time;
  device->status.low_time = device->status.device_time;
  return output;
}

Sync__::Sync__(std::string& prefix, std::string& name,
               std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {}

Tensor::Ptr Sync__::forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) {
  return input;
}

// --- Pipeline Parallelism P2P Communication ---

PipelineSend::PipelineSend(std::string& prefix, std::string& name,
                           int peer_device_rank,
                           std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, false),
      peer_device_rank(peer_device_rank) {
    // Register placeholder output tensor (shape updated in forward).
    std::vector<int> shape = {1, 1};
    Tensor::Ptr output = Tensor::Create("pipeline_activation", shape, "act",
                                         device, device->model_config.precision_byte);
    add_tensor(output);
}

Tensor::Ptr PipelineSend::forward(const Tensor::Ptr input,
                                   BatchedSequence::Ptr sequences_metadata) {
    Tensor::Ptr output = get_activation("pipeline_activation", input->shape);

    long size = input->getSize();
    if (size == 0) return output;

    // Select NVLink (intra-node) or InfiniBand (inter-node) based on
    // whether sender and receiver are on the same physical node.
    int src_node = device->device_total_rank / device->config.num_device;
    int dst_node = peer_device_rank / device->config.num_device;

    double latency, bandwidth;
    if (src_node == dst_node) {
        latency = device->config.device_ict_latency;
        bandwidth = device->config.device_ict_bandwidth;
    } else {
        latency = device->config.node_ict_latency;
        bandwidth = device->config.node_ict_bandwidth;
    }

    // Same unit conversion as AllReduce: size / bw * 1e9 + latency
    time_ns total_time = size / bandwidth * 1000 * 1000 * 1000
                         + latency;
    // TODO: Add parallel_execution + communication_hiding support
    // (matching AllReduce pattern with high_time/low_time branching)
    device->status.device_time += total_time;

    return output;
}

PipelineRecv::PipelineRecv(std::string& prefix, std::string& name,
                           int peer_device_rank,
                           std::string send_module_map_name,
                           std::string send_tensor_name,
                           std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true),  // sync=true
      peer_device_rank(peer_device_rank),
      send_module_map_name(send_module_map_name),
      send_tensor_name(send_tensor_name) {
    // Register placeholder output tensor.
    std::vector<int> shape = {1, 1};
    Tensor::Ptr output = Tensor::Create("pipeline_activation", shape, "act",
                                         device, device->model_config.precision_byte);
    add_tensor(output);
}

Tensor::Ptr PipelineRecv::forward(const Tensor::Ptr input,
                                   BatchedSequence::Ptr sequences_metadata) {
    // sync_devices() only touches dependency tensor devices (= sender stage),
    // NOT the local device. We must explicitly advance local time to model
    // the fill bubble (waiting for sender to complete).
    Tensor::Ptr send_tensor = device->cluster->find_tensor(
        peer_device_rank, send_module_map_name, send_tensor_name);
    time_ns send_time = send_tensor->get_device()->status.device_time;

    // Advance local device_time to at least sender's completion time.
    // Also align high_time/low_time following Sync__Set pattern (line 541-545).
    device->status.device_time = std::max(device->status.device_time, send_time);
    device->status.high_time = device->status.device_time;
    device->status.low_time = device->status.device_time;

    // Reconstruct shape from metadata (not trace-time dummy input).
    int process_tokens = sequences_metadata->get_process_token();
    int hidden_dim = device->model_config.hidden_dim;
    Tensor::Ptr output = get_activation("pipeline_activation",
        {process_tokens, hidden_dim});
    return output;
}

// Override the default dependency resolution to look up PipelineSend's
// output tensor on the peer device, rather than the standard same-module
// lookup across our own device_list.
void PipelineRecv::set_dependency_tensor(
    std::vector<Tensor::Ptr>& dependency_tensor_list, Tensor::Ptr tensor) {
    Tensor::Ptr send_output = device->cluster->find_tensor(
        peer_device_rank, send_module_map_name, send_tensor_name);
    dependency_tensor_list.clear();
    dependency_tensor_list.push_back(send_output);
}

}  // namespace llm_system

