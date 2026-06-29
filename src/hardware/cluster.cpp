#include "hardware/cluster.h"

#include <climits>
#include <filesystem>

#include "common/assert.h"
#include "hardware/stat.h"
#include "model/util.h"
#include "module/module_graph.h"
#include "module/tensor.h"
#include "module/timeboard.h"

namespace llm_system {

namespace {

double bytesToGB(hw_metric bytes) {
  return bytes / 1024.0 / 1024 / 1024;
}

struct MemoryCheckContext {
  int pp_dg = 1;
  int ne_tp_dg = 1;
  int e_tp_dg = 1;
  int dp_degree_mem = 1;
  int activation_batch_size_per_dp = 0;
  int activation_total_batch_size = 0;
  bool exclude_activation_for_hbf = false;
};

struct StageMemoryStats {
  int stage_index = 0;
  int stage_layers = 0;
  hw_metric activation_size = 0;
  hw_metric weight_size = 0;
  hw_metric cache_size = 0;
  hw_metric total_size = 0;
  hw_metric fixed_size = 0;
  hw_metric kv_per_seq = 0;
  bool raw_activation_buffer_exceeded = false;
};

MemoryCheckContext buildMemoryCheckContext(const Scheduler::Ptr& scheduler,
                                           int num_total_device,
                                           const SystemConfig& config) {
  MemoryCheckContext context;
  context.pp_dg = scheduler->model_config.pp_dg;
  context.ne_tp_dg = scheduler->model_config.ne_tp_dg;
  context.e_tp_dg = scheduler->model_config.e_tp_dg;
  context.dp_degree_mem = num_total_device / (context.ne_tp_dg * context.pp_dg);
  context.activation_batch_size_per_dp =
      (scheduler->batch_size_per_dp + context.pp_dg - 1) / context.pp_dg;
  context.activation_total_batch_size =
      context.activation_batch_size_per_dp * context.dp_degree_mem;
  context.exclude_activation_for_hbf = config.gpu_gen.rfind("HBF", 0) == 0;
  return context;
}

void resetSchedulerAfterMemoryCheck(const Scheduler::Ptr& scheduler) {
  scheduler->clear();
  scheduler->initRunningQueue();
}

hw_metric computeStageKvPerSeq(const Device::Ptr& device, int ls, int le) {
  if ((device->model_config.qk_rope_head_dim != 0) &&
      (device->model_config.compressed_kv == true)) {
    return 1.0 * (device->model_config.input_len + device->model_config.output_len) *
           (device->model_config.kv_lora_rank +
            device->model_config.qk_rope_head_dim) *
           (le - ls) * device->model_config.precision_byte;
  }

  long long full_len =
      device->model_config.input_len + device->model_config.output_len;
  long long total_effective_len = 0;
  for (int l = ls; l < le; l++) {
    total_effective_len += effective_kv_len(device->model_config, l, full_len);
  }
  return 2.0 * total_effective_len * device->model_config.head_dim *
         device->model_config.num_kv_heads / device->model_config.ne_tp_dg *
         device->model_config.precision_byte;
}

StageMemoryStats evaluateStageMemory(
    Cluster& cluster,
    const MemoryCheckContext& context,
    int pp) {
  StageMemoryStats stats;
  stats.stage_index = pp;

  int repr = pp * context.ne_tp_dg;
  Device::Ptr device = cluster.get_device(repr);
  auto module = cluster.module_map.at(repr).at("::LLM");
  auto size_vector = module->get_size();

  auto [ls, le] = get_stage_layer_range(device->model_config, pp);
  stats.stage_layers = le - ls;

  int stage_dev_count =
      get_stage_device_list(pp, context.ne_tp_dg, context.pp_dg,
                            context.dp_degree_mem)
          .size();
  int num_routed_expert_per_device =
      device->model_config.num_routed_expert * context.e_tp_dg /
      stage_dev_count;
  int expert_batch_size = device->model_config.expert_freq
                              ? context.activation_total_batch_size *
                                    device->model_config.top_k /
                                    device->model_config.num_routed_expert
                              : 0;

  int input_len = device->model_config.input_len;
  int total_len =
      device->model_config.input_len + device->model_config.output_len;
  int hidden_dim = device->model_config.hidden_dim;
  int q_lora_rank = device->model_config.q_lora_rank;
  int kv_lora_rank = device->model_config.kv_lora_rank;
  int qk_rope_head_dim = device->model_config.qk_rope_head_dim;
  int head_dim = device->model_config.head_dim;
  int num_heads = device->model_config.num_heads;
  int intermediate_dim = device->model_config.intermediate_dim;
  int ffn_way = device->model_config.ffn_way;
  int expert_intermediate_dim = device->model_config.expert_intermediate_dim;

  if (cluster.config.decode_mode) {
    if (device->model_config.use_absorb) {
      stats.activation_size =
          ((context.activation_batch_size_per_dp * hidden_dim) +
           (context.activation_batch_size_per_dp * q_lora_rank) +
           (context.activation_batch_size_per_dp * kv_lora_rank) +
           (context.activation_batch_size_per_dp * qk_rope_head_dim) +
           (context.activation_batch_size_per_dp *
            (3.0 * qk_rope_head_dim + head_dim) * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * num_heads * kv_lora_rank /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * 2.0 * num_heads *
            total_len / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * num_heads * kv_lora_rank /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * num_heads * head_dim /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * hidden_dim) +
           (num_routed_expert_per_device +
            device->model_config.num_shared_expert) *
               ((expert_batch_size * 2.0 * expert_intermediate_dim) +
                (expert_batch_size * expert_intermediate_dim) +
                (expert_batch_size * hidden_dim))) *
          device->model_config.precision_byte;
    } else if (device->model_config.compressed_kv) {
      stats.activation_size =
          ((context.activation_batch_size_per_dp * hidden_dim) +
           (context.activation_batch_size_per_dp * q_lora_rank) +
           (context.activation_batch_size_per_dp * kv_lora_rank) +
           (context.activation_batch_size_per_dp * qk_rope_head_dim) +
           (context.activation_batch_size_per_dp *
            (3.0 * qk_rope_head_dim + head_dim) * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * 2.0 * total_len * head_dim *
            num_heads / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * 2.0 * total_len * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * num_heads * head_dim /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * hidden_dim) +
           (num_routed_expert_per_device +
            device->model_config.num_shared_expert) *
               (2.0 * (expert_batch_size * expert_intermediate_dim) +
                (expert_batch_size * expert_intermediate_dim) +
                (expert_batch_size * hidden_dim))) *
          device->model_config.precision_byte;
    } else {
      stats.activation_size =
          ((context.activation_batch_size_per_dp * hidden_dim) +
           (context.activation_batch_size_per_dp * q_lora_rank) +
           (context.activation_batch_size_per_dp * kv_lora_rank) +
           (context.activation_batch_size_per_dp * qk_rope_head_dim) +
           (context.activation_batch_size_per_dp *
            (3.0 * qk_rope_head_dim + head_dim) * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * 2.0 * head_dim * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * 2.0 * total_len * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * num_heads * head_dim /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * hidden_dim) +
           (num_routed_expert_per_device +
            device->model_config.num_shared_expert) *
               (2.0 * (expert_batch_size * expert_intermediate_dim) +
                (expert_batch_size * expert_intermediate_dim) +
                (expert_batch_size * hidden_dim))) *
          device->model_config.precision_byte;
    }
  } else {
    if (device->model_config.use_absorb) {
      stats.activation_size =
          ((context.activation_batch_size_per_dp * input_len * hidden_dim) +
           (context.activation_batch_size_per_dp * input_len * q_lora_rank) +
           (context.activation_batch_size_per_dp * input_len * kv_lora_rank) +
           (context.activation_batch_size_per_dp * input_len *
            qk_rope_head_dim) +
           (context.activation_batch_size_per_dp * input_len *
            (3.0 * qk_rope_head_dim + head_dim) * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * num_heads *
            kv_lora_rank / context.ne_tp_dg) +
           2.0 * (context.activation_batch_size_per_dp * input_len * num_heads *
                  input_len / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * num_heads *
            kv_lora_rank / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * num_heads *
            head_dim / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * hidden_dim) +
           (num_routed_expert_per_device +
            device->model_config.num_shared_expert) *
               (2.0 * (expert_batch_size * input_len *
                       expert_intermediate_dim) +
                (expert_batch_size * input_len * expert_intermediate_dim) +
                (expert_batch_size * input_len * hidden_dim))) *
          device->model_config.precision_byte;
    } else {
      stats.activation_size =
          ((context.activation_batch_size_per_dp * input_len * hidden_dim) +
           (context.activation_batch_size_per_dp * input_len * q_lora_rank) +
           (context.activation_batch_size_per_dp * input_len * kv_lora_rank) +
           (context.activation_batch_size_per_dp * input_len *
            qk_rope_head_dim) +
           (context.activation_batch_size_per_dp * input_len *
            (3.0 * qk_rope_head_dim + head_dim) * num_heads /
            context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * 2.0 * head_dim *
            num_heads / context.ne_tp_dg) +
           2.0 * (context.activation_batch_size_per_dp * input_len * input_len *
                  num_heads / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * num_heads *
            head_dim / context.ne_tp_dg) +
           (context.activation_batch_size_per_dp * input_len * hidden_dim) +
           (num_routed_expert_per_device +
            device->model_config.num_shared_expert) *
               (2.0 * (expert_batch_size * input_len *
                       expert_intermediate_dim) +
                (expert_batch_size * input_len * expert_intermediate_dim) +
                (expert_batch_size * input_len * hidden_dim))) *
          device->model_config.precision_byte;
    }
  }

  if (device->model_config.q_lora_rank == 0) {
    if (cluster.config.decode_mode && cluster.config.disagg_system &&
        cluster.config.use_flash_attention) {
      hw_metric activation_size_dense =
          ((context.activation_batch_size_per_dp * hidden_dim) +
           (context.activation_batch_size_per_dp * hidden_dim) +
           (context.activation_batch_size_per_dp * intermediate_dim *
            (ffn_way - 1) / context.ne_tp_dg)) *
          device->model_config.precision_byte;
      hw_metric activation_size_expert =
          ((expert_batch_size * hidden_dim) +
           (expert_batch_size * hidden_dim) +
           (expert_batch_size * hidden_dim * 
            (context.dp_degree_mem - 1) / context.dp_degree_mem) +
           (expert_batch_size * expert_intermediate_dim *
            (ffn_way - 1) / context.e_tp_dg)) *
          device->model_config.precision_byte;
      hw_metric activation_size_LMhead_before_AR =
          context.activation_batch_size_per_dp *
          (hidden_dim + device->model_config.n_vocab / context.ne_tp_dg) *
          device->model_config.precision_byte;
      hw_metric activation_size_LMhead_after_AR =
          context.activation_batch_size_per_dp * device->model_config.n_vocab *
          device->model_config.precision_byte;
      hw_metric activation_size_decoder_block =
          std::max(activation_size_dense, activation_size_expert);
      hw_metric activation_size_LM_head =
          std::max(activation_size_LMhead_before_AR,
                   activation_size_LMhead_after_AR);
      stats.activation_size =
          std::max(activation_size_decoder_block, activation_size_LM_head);
    } else {
      stats.activation_size = size_vector.at(0) / stats.stage_layers;
    }
  }

  stats.weight_size = size_vector.at(1);
  stats.cache_size = size_vector.at(2);
  if (context.exclude_activation_for_hbf) {
    stats.total_size = stats.weight_size + stats.cache_size;
    stats.fixed_size = stats.weight_size;
  } else {
    stats.total_size = stats.activation_size + stats.weight_size + stats.cache_size;
    stats.fixed_size = stats.activation_size + stats.weight_size;
  }

  stats.raw_activation_buffer_exceeded =
      context.exclude_activation_for_hbf &&
      stats.activation_size > cluster.config.intermediate_buffer;
  stats.kv_per_seq = computeStageKvPerSeq(device, ls, le);
  return stats;
}

void reportStageMemoryStats(const StageMemoryStats& stats,
                            const SystemConfig& config) {
  std::cout << "Stage " << stats.stage_index << " ACT: "
            << bytesToGB(stats.activation_size)
            << "GB, Weight: " << bytesToGB(stats.weight_size)
            << "GB, Cache: " << bytesToGB(stats.cache_size) << "GB"
            << std::endl;

  if (stats.raw_activation_buffer_exceeded) {
    std::cout << "Stage " << stats.stage_index;
    if (config.ignore_activation_buffer_oom) {
      std::cout << " activation buffer limit exceeded (ignored): ACT ";
    } else {
      std::cout << " activation buffer OOM: ACT ";
    }
    std::cout << bytesToGB(stats.activation_size)
              << "GB > intermediate_buffer "
              << bytesToGB(config.intermediate_buffer) << "GB" << std::endl;
  }

  std::cout << "Stage " << stats.stage_index << " Total: "
            << bytesToGB(stats.total_size) << "GB" << std::endl;
}

int computeMemCapLimitedBatch(const SystemConfig& config,
                              const Scheduler::Ptr& scheduler,
                              const std::vector<StageMemoryStats>& stage_stats) {
  int min_batch = INT_MAX;
  for (const auto& stats : stage_stats) {
    hw_metric avail = config.memory_capacity - stats.fixed_size;
    if (avail < 0) {
      fail("Stage " + std::to_string(stats.stage_index) +
           ": memory capacity smaller than model weight");
    }
    if (stats.kv_per_seq > 0) {
      int stage_batch = static_cast<int>(avail / stats.kv_per_seq) *
                        scheduler->dp_degree;
      std::cout << "Stage " << stats.stage_index << ": avail="
                << bytesToGB(avail) << "GB, kv/seq="
                << bytesToGB(stats.kv_per_seq) << "GB, max_batch="
                << stage_batch << std::endl;
      min_batch = std::min(min_batch, stage_batch);
    }
  }
  return min_batch - 1;
}

}  // namespace

Cluster::Cluster(SystemConfig config, Scheduler::Ptr scheduler)
    : config(config), scheduler(scheduler), executor() {
  cluster_ict_latency = config.node_ict_latency;
  cluster_ict_bandwidth = config.node_ict_bandwidth;
  num_device = config.num_device;
  num_node = config.num_node;
  num_total_device = num_device * num_node;
}

Device::Ptr Cluster::get_device(int device_total_rank) {
  int node_id = device_total_rank / num_device;
  return node.at(node_id)->get_device(device_total_rank);
}

void Cluster::add_module(int device_rank, std::string name,
                         Module::Ptr module) {
  auto &module_map_ = module_map.at(device_rank);

  if (module_map_.find(name) == module_map_.end()) {
    module_map_.emplace(name, module);
  } else {
    fail("Cluster::add_module, same module name");
  }
}

void Cluster::set_dependency() {
  for (Node::Ptr _node : node) {
    _node->set_dependency();
  }
}

void Cluster::restartModuleGraph() {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    device->restartGraph();
    device->reset_status();
    device->reset_timeboard();
  }
}

void Cluster::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    device->initializeDRAM(ProcessorType, dramEnergy);
  }
}

Tensor::Ptr Cluster::find_tensor(int device_rank,
                                  const std::string& module_map_name,
                                  const std::string& tensor_name) {
  Module::Ptr module = module_map.at(device_rank).at(module_map_name);
  return module->get_tensor(tensor_name);
}

void Cluster::set_dependency_tensor(std::vector<Tensor::Ptr> &list,
                                    Tensor::Ptr tensor,
                                    const std::vector<int> &device_list) {
  list.resize(0);

  Tensor::Ptr temp;
  Module::Ptr module;
  for (int device_rank :
       device_list) {
    module = module_map.at(device_rank).at(tensor->get_module_map_name());
    temp = module->get_activation(tensor->name, {}, false);
    list.push_back(temp);
  }
}

bool Cluster::checkMemorySize() {
  out_of_memory = false;

  MemoryCheckContext context =
      buildMemoryCheckContext(scheduler, num_total_device, config);
  std::vector<StageMemoryStats> stage_stats;
  stage_stats.reserve(context.pp_dg);

  const StageMemoryStats* worst_total_stage = nullptr;
  const StageMemoryStats* worst_activation_stage = nullptr;

  for (int pp = 0; pp < context.pp_dg; pp++) {
    stage_stats.push_back(evaluateStageMemory(*this, context, pp));
    const StageMemoryStats& stats = stage_stats.back();
    reportStageMemoryStats(stats, config);

    if (worst_total_stage == nullptr ||
        stats.total_size > worst_total_stage->total_size) {
      worst_total_stage = &stats;
    }
    if (stats.raw_activation_buffer_exceeded &&
        (worst_activation_stage == nullptr ||
         stats.activation_size > worst_activation_stage->activation_size)) {
      worst_activation_stage = &stats;
    }
  }

  bool memory_capacity_oom =
      worst_total_stage != nullptr &&
      worst_total_stage->total_size > config.memory_capacity;
  bool raw_activation_buffer_oom = worst_activation_stage != nullptr;
  bool effective_activation_buffer_oom =
      raw_activation_buffer_oom && !config.ignore_activation_buffer_oom;

  if (memory_capacity_oom) {
    std::cout << "Memory capacity OOM at stage "
              << worst_total_stage->stage_index << ": total "
              << bytesToGB(worst_total_stage->total_size)
              << "GB > memory_capacity "
              << bytesToGB(config.memory_capacity) << "GB" << std::endl;
  }
  if (raw_activation_buffer_oom) {
    if (config.ignore_activation_buffer_oom) {
      std::cout << "Activation buffer limit exceeded (ignored) at stage ";
    } else {
      std::cout << "Activation buffer OOM at stage ";
    }
    std::cout << worst_activation_stage->stage_index << ": ACT "
              << bytesToGB(worst_activation_stage->activation_size)
              << "GB > intermediate_buffer "
              << bytesToGB(config.intermediate_buffer) << "GB" << std::endl;
  }

  if (!memory_capacity_oom && !effective_activation_buffer_oom) {
    return false;
  }

  out_of_memory = true;
  if (config.exit_out_of_memory) {
    return true;
  }
  if (memory_capacity_oom && !effective_activation_buffer_oom &&
      config.mem_cap_limit == true) {
    int max_batch_size = computeMemCapLimitedBatch(config, scheduler, stage_stats);
    std::cout << "Modify max_batch_size to " << max_batch_size << std::endl;
    scheduler->total_batch_size = max_batch_size;
    scheduler->batch_size_per_dp = max_batch_size / scheduler->dp_degree;
    resetSchedulerAfterMemoryCheck(scheduler);
    return false;
  }

  resetSchedulerAfterMemoryCheck(scheduler);
  return false;
}

bool Cluster::checkHeteroMemorySize() {
  Device::Ptr device = get_device(0);
  auto module = module_map.at(0).at("::LLM");
  auto size_vector = module->get_size();

  std::cout << "ACT: "
            << size_vector.at(0) / 1024.0 / 1024 / 1024 /
                   device->model_config.num_layers
            << "GB, Weight: " << size_vector.at(1) / 1024.0 / 1024 / 1024
            << "GB, Cache: " << size_vector.at(2) / 1024.0 / 1024 / 1024 << "GB"
            << std::endl;
  double size = size_vector.at(1) + size_vector.at(2) - 3.3 * 1024.0 * 1024.0 * 1024.0 /
                device->model_config.ne_tp_dg; // Non MoE weight
  std::cout << "Total: " << size / 1024.0 / 1024 / 1024  << "GB" << std::endl;
  if (size > config.memory_capacity) {
    if (config.exit_out_of_memory) {
      return true;
    } else {
      // Chunked attention: sum effective KV length across layers
      long long full_len = device->model_config.input_len + device->model_config.output_len;
      long long total_effective_len = 0;
      for (int l = 0; l < device->model_config.num_layers; l++) {
        total_effective_len += effective_kv_len(device->model_config, l, full_len);
      }
      long kv_cache_size_per_seq =
          2 * total_effective_len * device->model_config.head_dim *
          device->model_config.num_kv_heads / device->model_config.ne_tp_dg *
          device->model_config.precision_byte;

      hw_metric avail_capacity = config.memory_capacity - (size_vector.at(0) / device->model_config.num_layers) -
        size_vector.at(1);
      if (avail_capacity < 0) {
        fail("Memory capacity is smaller than model weight");
      }
      std::cout << "Available capacity for KV cache is "
                << avail_capacity / 1024.0 / 1024 / 1024 << "GB" << std::endl;
      std::cout << "KV cache per seq is "
                << kv_cache_size_per_seq / 1024.0 / 1024 / 1024 << "GB" << std::endl;                
      int max_batch_size =
          (int)(avail_capacity / kv_cache_size_per_seq) * scheduler->dp_degree;
      std::cout << "Modify max_batch_size to " << max_batch_size - 1
                << std::endl;
      scheduler->total_batch_size = max_batch_size - 1;
      scheduler->batch_size_per_dp =
          (max_batch_size - 1) / scheduler->dp_degree;
      resetSchedulerAfterMemoryCheck(scheduler);
      return false;
    }
  }
  return false;
}

std::vector<energy_nJ> Cluster::getTotalEnergy() {
  std::vector<energy_nJ> total_energy = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    std::vector<energy_nJ> device_energy =
        device->top_module_graph->getDeviceEnergy();
    for (int e_idx = 0; e_idx < total_energy.size(); e_idx++) {
      total_energy[e_idx] += device_energy[e_idx];
    }
  }
  return total_energy;
}

bool Cluster::check_module_graph_remain() {
  for (Node::Ptr _node : node) {
    if (_node->check_module_graph_remain()) {
      return true;
    }
  }
  return false;
}

void Cluster::exportToCSV(std::ofstream &csv, std::vector<Stat> &stat_list) {
  for (auto temp : stat_list) {
    csv << std::to_string(temp.iter_info) << "," << std::to_string(temp.split)
        << "," << temp.type << "," << std::to_string(temp.time) << ","
        << std::to_string(temp.latency) << ","
        << std::to_string(temp.queueing_delay) << ","
        << std::to_string(temp.arrival_time) << ","
        << std::to_string(temp.seq_queue_size) << ","
        << std::to_string(temp.input_len) << ","
        << std::to_string(temp.output_len) << ","
        << std::to_string(temp.num_sum_iter) << ","
        << std::to_string(temp.is_mixed) << ","
        << std::to_string(temp.batchsize) << ","
        << std::to_string(temp.process_token) << ","
        << std::to_string(temp.sum_seq) << "," << std::to_string(temp.gen_seq)
        << "," << std::to_string(temp.average_seq_len) << ","
        << std::to_string(temp.sum_attention_opb) << ","
        << std::to_string(temp.qkv_gen) << "," 
        << std::to_string(temp.q_down_proj) << "," 
        << std::to_string(temp.kv_down_proj) << ","
        << std::to_string(temp.kr_proj) << ","
        << std::to_string(temp.q_up_proj) << ","
        << std::to_string(temp.qr_proj) << ","
        << std::to_string(temp.kv_up_proj) << ","
        << std::to_string(temp.tr_k_up_proj) << ","
        << std::to_string(temp.v_up_proj) << ","
        << std::to_string(temp.atten_sum)
        << "," << std::to_string(temp.atten_gen) << ","
        << std::to_string(temp.o_proj) << ","
        << std::to_string(temp.lm_head) << ","
        << std::to_string(temp.ffn) << ","
        << std::to_string(temp.expert_ffn) << ","
        << std::to_string(temp.communication) << ","
        << std::to_string(temp.decode_kv_write) << ","
        << std::to_string(temp.pipeline_recv_wait) << ","
        << std::to_string(temp.rope) << ","
        << std::to_string(temp.layernorm) << ","
        << std::to_string(temp.residual) << ","
        << std::to_string(temp.act_energy) << ","
        << std::to_string(temp.read_energy) << ","
        << std::to_string(temp.write_energy) << ","
        << std::to_string(temp.all_act_energy) << ","
        << std::to_string(temp.all_read_energy) << ","
        << std::to_string(temp.all_write_energy) << ","
        << std::to_string(temp.mac_energy) << ","
        << std::to_string(temp.total_energy) << ","
        << std::to_string(temp.FC_DRAM_energy) << ","
        << std::to_string(temp.FC_COMP_energy) << ","
        << std::to_string(temp.Attn_DRAM_energy) << ","
        << std::to_string(temp.Attn_COMP_energy) << ","
        << std::to_string(temp.MoE_DRAM_energy) << ","
        << std::to_string(temp.MoE_COMP_energy) << ","
        << std::to_string(temp.isOOM) << ","
        << std::to_string(temp.second_token_time) << ","
        << std::to_string(temp.token_interval) << ","
        << std::to_string(temp.generated_tokens) << ","
        << std::to_string(temp.cumulative_generated_tokens) << ","
        << std::to_string(temp.tps) << ","
        << std::to_string(temp.tps_per_gpu) << ","
        << std::to_string(temp.seeded_remaining) << ","
        << std::to_string(temp.first_departure_seen) << ","
        << std::to_string(temp.post_departure_steps)
        << "," << std::to_string(temp.three_year_pec) << std::endl;
  }
  stat_list.resize(0);
}

// --- Three-Year PEC (Program/Erase Count) helpers ---
//
// PEC estimates how many times the entire HBF memory capacity would be
// overwritten by KV-cache writes if the current write rate continued for
// 3 years.  Only non-MLA KV writes are tracked (QKV projection KV output
// in linear_impl.cpp and bulk decode KV writes in decode_kv_write.cpp).
//
// Formula:
//   three_year_pec = max_i(delta_kv_write_bytes_i) / memory_capacity
//                    * (three_year_ns / elapsed_sim_time_ns)
//
// Key invariant: per-device deltas are computed first, then max is taken.
// This avoids errors when the max-write device changes between baseline
// and the current snapshot.

// Reset all PEC tracking state at the start of each simulation run.
// Called from runIteration() so that consecutive runs in the same process
// do not carry over stale cumulative write counts or baseline snapshots.
void Cluster::resetPECState() {
  pec_baseline_initialized = false;
  pec_start_time = 0;
  pec_baseline_kv_write_bytes.clear();
  for (int d = 0; d < num_total_device; d++) {
    get_device(d)->cumulative_kv_write_bytes = 0;
  }
}

// Compute three-year PEC for the current stat row.
// - Returns 0 for non-HBF GPUs (no HBF wear tracking needed).
// - On the first t2t row, snapshots baseline time and per-device cumulative
//   KV write bytes, then returns 0 (warmup write excluded).
// - On subsequent rows, computes running PEC from per-device write deltas.
// - Non-t2t rows before the first t2t baseline also return 0.
double Cluster::computeThreeYearPEC(time_ns current_time, const std::string& type) {
  // non-HBF: PEC is always 0
  if (config.gpu_gen.rfind("HBF", 0) != 0) return 0.0;

  // Baseline is captured on the first t2t (decode) row only.
  // Sum/t2ft/e2e rows before the first t2t get PEC = 0.
  if (!pec_baseline_initialized) {
    if (type != "t2t") return 0.0;
    pec_baseline_initialized = true;
    pec_start_time = current_time;
    pec_baseline_kv_write_bytes.resize(num_total_device);
    for (int d = 0; d < num_total_device; d++) {
      pec_baseline_kv_write_bytes[d] = get_device(d)->cumulative_kv_write_bytes;
    }
    return 0.0;  // first t2t row itself is 0
  }

  time_ns elapsed = current_time - pec_start_time;
  if (elapsed <= 0) return 0.0;

  // Compute per-device delta from baseline, then take max.
  // Must NOT do max(current) - max(baseline) — the max device may differ.
  long long max_delta = 0;
  for (int d = 0; d < num_total_device; d++) {
    long long delta = get_device(d)->cumulative_kv_write_bytes
                      - pec_baseline_kv_write_bytes[d];
    if (delta > max_delta) max_delta = delta;
  }

  constexpr double three_year_ns = 3.0 * 365.0 * 24.0 * 3600.0 * 1e9;
  double pec = (double)max_delta / (double)config.memory_capacity
               * (three_year_ns / (double)elapsed);
  return pec;
}

// Populate stat.three_year_pec before pushing to stat_list.
// Called immediately before every stat_list.push_back(stat).
void Cluster::finalizeThreeYearPEC(Stat& stat) {
  if (stat.type != "t2t") {
    stat.three_year_pec = 0.0;
    return;
  }
  stat.three_year_pec = computeThreeYearPEC(stat.time, stat.type);
}

// CSV export without energy columns.
// To restore energy output, replace exportToCSVNoEnergy() with exportToCSV()
// and switch the header back to the full version in runIteration().
void Cluster::exportToCSVNoEnergy(std::ofstream &csv, std::vector<Stat> &stat_list) {
  for (auto temp : stat_list) {
    csv << std::to_string(temp.iter_info) << "," << std::to_string(temp.split)
        << "," << temp.type << "," << std::to_string(temp.time) << ","
        << std::to_string(temp.latency) << ","
        << std::to_string(temp.queueing_delay) << ","
        << std::to_string(temp.arrival_time) << ","
        << std::to_string(temp.seq_queue_size) << ","
        << std::to_string(temp.input_len) << ","
        << std::to_string(temp.output_len) << ","
        << std::to_string(temp.num_sum_iter) << ","
        << std::to_string(temp.is_mixed) << ","
        << std::to_string(temp.batchsize) << ","
        << std::to_string(temp.process_token) << ","
        << std::to_string(temp.sum_seq) << "," << std::to_string(temp.gen_seq)
        << "," << std::to_string(temp.average_seq_len) << ","
        << std::to_string(temp.sum_attention_opb) << ","
        << std::to_string(temp.qkv_gen) << ","
        << std::to_string(temp.q_down_proj) << ","
        << std::to_string(temp.kv_down_proj) << ","
        << std::to_string(temp.kr_proj) << ","
        << std::to_string(temp.q_up_proj) << ","
        << std::to_string(temp.qr_proj) << ","
        << std::to_string(temp.kv_up_proj) << ","
        << std::to_string(temp.tr_k_up_proj) << ","
        << std::to_string(temp.v_up_proj) << ","
        << std::to_string(temp.atten_sum)
        << "," << std::to_string(temp.atten_gen) << ","
        << std::to_string(temp.o_proj) << ","
        << std::to_string(temp.lm_head) << ","
        << std::to_string(temp.ffn) << ","
        << std::to_string(temp.expert_ffn) << ","
        << std::to_string(temp.communication) << ","
        << std::to_string(temp.decode_kv_write) << ","
        << std::to_string(temp.pipeline_recv_wait) << ","
        << std::to_string(temp.rope) << ","
        << std::to_string(temp.layernorm) << ","
        << std::to_string(temp.residual) << ","
        << std::to_string(temp.isOOM) << ","
        << std::to_string(temp.second_token_time) << ","
        << std::to_string(temp.token_interval) << ","
        << std::to_string(temp.generated_tokens) << ","
        << std::to_string(temp.cumulative_generated_tokens) << ","
        << std::to_string(temp.tps) << ","
        << std::to_string(temp.tps_per_gpu) << ","
        << std::to_string(temp.seeded_remaining) << ","
        << std::to_string(temp.first_departure_seen) << ","
        << std::to_string(temp.post_departure_steps)
        << "," << std::to_string(temp.three_year_pec) << std::endl;
  }
  stat_list.resize(0);
}

std::vector<Stat> Cluster::runIteration(int iter, std::string file_name) {
  std::ofstream csv;
  csv.open(file_name);

  // Reset PEC tracking so consecutive runs don't carry over stale state.
  resetPECState();

  // CSV header without energy columns.
  // To restore energy columns, use the full header with act_energy..moe_comp
  // and replace exportToCSVNoEnergy() calls with exportToCSV().
  csv << "iter_info,split,type,time,latency,queueing_delay,arrival_time,seq_queue_"
         "size,"
         "input_len,output_len,num_sum_iter,mixed,batchsize,numtoken,num_sum_"
         "seq,num_gen_seq,seqlen,sum_attention_opb,qkvgen,q_down_proj,kv_down_proj,kr_proj,"
         "q_up_proj,qr_proj,kv_up_proj,tr_k_up_proj,v_up_proj,atten_sum,atten_gen,"
         "o_proj,lm_head,ffn,expert_ffn,communication,"
         "decode_kv_write,pipeline_recv_wait,"
         "rope,layernorm,residual,OOM,"
         "second_token_time,token_interval,"
         "generated_tokens,cumulative_generated_tokens,tps,tps_per_gpu,seeded_remaining,"
         "first_departure_seen,post_departure_steps,three_year_pec"
      << std::endl;

  std::vector<Stat> stat_list;

  bool skip_hitting_queue = config.disagg_system && config.decode_mode && config.single_trace_steady_state;
  scheduler->fillSequenceQueue();
  scheduler->fillRunningQueue();

  if (!skip_hitting_queue)
    scheduler->hittingQueue(10000);

  // dowon revised
  // Execution routing: PP>1 decode uses the pipeline-decode
  // path regardless of disagg_system. PP=1 uses existing paths.
  if (scheduler->model_config.pp_dg > 1) {
    assertTrue(config.decode_mode,
              "pipeline parallelism with PP>1 is currently only supported for decode-mode models");
    stat_list = runIterationPipelineDecode(iter, csv);
  } else if (config.disagg_system) {
    stat_list = runIterationSumGenSplit(iter, csv);
  } else {
    stat_list = runIterationMixed(iter, csv);
  }

  // Export remaining stats that weren't written inside the iteration loop
  exportToCSVNoEnergy(csv, stat_list);

  std::cout << "Total: " << std::to_string(scheduler->total_time) << std::endl;
  std::cout << file_name << std::endl;
  csv.close();

  return stat_list;
}

std::vector<Stat> Cluster::runIterationMixed(int iter, std::ofstream &csv) {
  // dowon revised
  // Note: PP>1 decode now routes to runIterationPipelineDecode from runIteration().
  // This function only handles PP=1 or non-decode modes.
  constexpr int kCSVExportInterval = 25;

  time_ns total_time = 0;

  std::vector<Stat> stat_list;
  bool is_after_sum = false;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iter; i++) {
    if (i % kCSVExportInterval == kCSVExportInterval - 1) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - start);
      start = std::chrono::high_resolution_clock::now();
      exportToCSVNoEnergy(csv, stat_list);
    }

    auto metadata = scheduler->setMetadata();

    time_ns time;
    std::vector<Sequence::Ptr> token_list;

    run(metadata);
    time = 0;
    for (int d = 0; d < num_total_device; d++) {
      time = std::max(time, get_device(d)->status.device_time);
    }

    if (scheduler->getNumProcessToken() == 0) {
      time = 20 * 1000 * 1000;
      continue;
    }

    total_time += time;

    Stat stat;
    stat.iter_info = 1;
    stat.type = "t2t";
    stat.time = total_time;
    scheduler->total_time = total_time;
    if (config.disagg_system) { stat.split = 1; }

    std::vector<energy_nJ> total_energy = getTotalEnergy();
    stat.act_energy = total_energy[0];
    stat.read_energy = total_energy[1];
    stat.write_energy = total_energy[2];
    stat.all_act_energy = total_energy[3];
    stat.all_read_energy = total_energy[4];
    stat.all_write_energy = total_energy[5];
    stat.mac_energy = total_energy[6];
    stat.total_energy = total_energy[7];
    stat.seq_queue_size = scheduler->sequence_queue.size();

    setStat(stat);
    setTimeBreakDown(stat);

    finalizeThreeYearPEC(stat);
    stat_list.push_back(stat);
    token_list = scheduler->updateScheduler(time);
    

    addLatency(stat_list, token_list, total_time);
    scheduler->fillSequenceQueue(time, total_time);
    scheduler->fillRunningQueue();
  }

  return stat_list;
}

std::vector<Stat> Cluster::runIterationSumGenSplit(int iter,
                                                   std::ofstream &csv) {
  time_ns total_time = 0;

  time_ns sum_machine_time = 0;

  std::vector<Stat> stat_list;
  std::vector<Sequence::Ptr> token_list;

  time_ns gen_start_time = 0;
  auto start = std::chrono::high_resolution_clock::now();

  // Disagg decode-only sanity check: no sum sequences expected
  if (config.decode_mode) {
    assertTrue(!scheduler->hasSumSeq(),
        "decode_mode + disagg: no sum sequences expected");
  }

  // Single trace steady state: seed output progress in running_queue
  if (config.single_trace_steady_state) {
    scheduler->seedSteadyStateBatch();
    seeded_remaining = scheduler->countSeededSequences();
    std::cerr << "Single-trace steady state: seeded " << seeded_remaining
              << " sequences (PP=1 disagg)" << std::endl;
  }

  for (int i = 0; i < iter; i++) {
    // export to csv
    if (i % 25 == 24) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - start);
      start = std::chrono::high_resolution_clock::now();
      exportToCSVNoEnergy(csv, stat_list);
    }

    auto metadata = scheduler->setMetadata();
    run(metadata);
    
    // With PP, iteration ends when the slowest device finishes.
    time_ns time = 0;
    for (int d = 0; d < num_total_device; d++) {
      time = std::max(time, get_device(d)->status.device_time);
    }

    // if no reqeusts, add time
    if (scheduler->getNumProcessToken() == 0) {
      time = 20 * 1000 * 1000;
      continue;
    }

    // gen machine
    if (!scheduler->hasSumSeq()) {
      total_time += time;

      Stat stat;
      stat.iter_info = 1;
      stat.type = "t2t";
      stat.time = total_time;
      scheduler->total_time = total_time;

      // power
      std::vector<energy_nJ> total_energy = getTotalEnergy();
      stat.act_energy = total_energy[0];
      stat.read_energy = total_energy[1];
      stat.write_energy = total_energy[2];
      stat.all_act_energy = total_energy[3];
      stat.all_read_energy = total_energy[4];
      stat.all_write_energy = total_energy[5];
      stat.mac_energy = total_energy[6];
      stat.total_energy = total_energy[7];
      stat.seq_queue_size = scheduler->sequence_queue.size();

      setStat(stat);
      setTimeBreakDown(stat);

      // Single trace steady state: TPS tracking (PP=1 disagg path)
      if (config.single_trace_steady_state) {
        if (i == 0) {
          tps_started = false;  // warmup round
        } else if (!tps_started) {
          tps_started = true;
        }

        if (tps_started && !tps_finished) {
          int iter_gen = scheduler->getNumProcessToken();
          cumulative_generated_tokens += iter_gen;
          cumulative_elapsed_time += time;
          stat.generated_tokens = iter_gen;
        }
      }

      finalizeThreeYearPEC(stat);
      stat_list.push_back(stat);
      token_list = scheduler->updateScheduler(time);

      // Track seeded turnover only. Early termination is intentionally disabled
      // so single_trace_steady_state respects the user-provided iter value.
      if (config.single_trace_steady_state) {
        for (auto& seq : token_list) {
          if (seq->turnover_seed && seq->current_len >= seq->total_len) {
            seeded_remaining--;
          }
        }

        // Fill stat fields
        stat_list.back().cumulative_generated_tokens = cumulative_generated_tokens;
        if (cumulative_elapsed_time > 0 && tps_started) {
          stat_list.back().tps = (double)cumulative_generated_tokens * 1e9 / cumulative_elapsed_time;
          stat_list.back().tps_per_gpu = stat_list.back().tps / num_total_device;
        }
        stat_list.back().seeded_remaining = seeded_remaining;
        stat_list.back().first_departure_seen = first_seeded_departure_seen ? 1 : 0;
        stat_list.back().post_departure_steps = post_departure_decode_steps;
      }

      addLatency(stat_list, token_list, total_time);

      scheduler->fillSequenceQueue(time, total_time);
      scheduler->fillRunningQueue(sum_machine_time);
    }
    // sum machine
    else {
      Stat stat;
      stat.iter_info = 1;
      stat.type = "sum";
      stat.time = std::max(total_time, sum_machine_time) + time;
      stat.latency = time;
      finalizeThreeYearPEC(stat);
      stat_list.push_back(stat);

      sum_machine_time = stat.time;
      // tokens which generated first token or eos token
      token_list = scheduler->updateSchedulerSumGenSplit(time);
      addLatency(stat_list, token_list, stat.time);
    }
  }

  return stat_list;
}

void Cluster::addLatency(std::vector<Stat> &stat_list,
                         const std::vector<Sequence::Ptr> &seq_list,
                         time_ns time) {
  // dowon revised
  // Relax e2e filter for modes where first_token_time / arrival_time may be 0:
  // 1. Disagg decode-only: no prefill, initial batch has arrival_time=0
  // 2. PP decode microbatch mode: end_token_time is absolute, not accumulated
  bool disagg_decode_only = config.disagg_system && config.decode_mode;
  // PP decode microbatch mode: PP>1 + decode (disagg-independent)
  bool pp_decode_microbatch_mode = (scheduler->model_config.pp_dg > 1)
      && config.decode_mode;
  bool relax_e2e_filter = disagg_decode_only || pp_decode_microbatch_mode;

  for (auto &seq : seq_list) {
    seq->gen_start_time = time;

    // Suppress latency rows for seeded steady-state sequences.
    // These are warmup batch members — only used for TPS observation.
    if (seq->suppress_latency) continue;

    // e2e milestone: sequence completed
    if (seq->current_len == seq->total_len) {
      // dowon revised
      // In disagg decode-only or PP decode microbatch mode, allow e2e with zero first_token/arrival
      if (!relax_e2e_filter) {
        if (seq->first_token_time == 0.0 || seq->arrival_time == 0.0) continue;
      }

      Stat stat;
      stat.iter_info = 0;
      stat.time = time;
      stat.type = "e2e";
      stat.latency = seq->end_token_time;
      stat.input_len = seq->input_len;
      stat.output_len = seq->output_len;
      stat.num_sum_iter = seq->num_sum_iter;
      stat.queueing_delay = seq->queueing_delay;
      stat.arrival_time = seq->arrival_time;
      // Disagg decode-only metrics
      stat.second_token_time = seq->second_token_time;
      if (seq->output_len > 2 && seq->second_token_time > 0) {
        stat.token_interval = (seq->end_token_time - seq->second_token_time)
                              / (seq->output_len - 2);
      }
      finalizeThreeYearPEC(stat);
      stat_list.push_back(stat);

    } else if (seq->current_len == seq->input_len) {
      // t2ft milestone: first token generated (original filter preserved)
      if (seq->first_token_time == 0.0 || seq->arrival_time == 0.0) continue;

      Stat stat;
      stat.iter_info = 0;
      stat.time = time;
      stat.type = "t2ft";
      stat.latency = seq->first_token_time;
      stat.input_len = seq->input_len;
      stat.num_sum_iter = seq->num_sum_iter;
      stat.queueing_delay = seq->queueing_delay;
      stat.arrival_time = seq->arrival_time;
      finalizeThreeYearPEC(stat);
      stat_list.push_back(stat);
    }
  }
}

void Cluster::exportGantt(std::string gantt_file_path) {
  std::filesystem::path dir = gantt_file_path;
  std::filesystem::create_directories(dir);

  if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      std::filesystem::remove_all(entry);
    }
  } else {
    std::cerr << "Error: Directory does not exist.\n";
  }

  for (int i = 0; i < num_total_device; i++) {
    TopModuleGraph::Ptr top = get_device(i)->top_module_graph;
    top->exportGantt(gantt_file_path, i);
  }
}
void Cluster::setStat(Stat &stat) {
  
  // With PP, use max device time across all devices.
  time_ns time = 0;
  for (int d = 0; d < num_total_device; d++) {
    time = std::max(time, get_device(d)->status.device_time);
  }

  stat.batchsize = scheduler->getBatchSize();
  stat.average_seq_len = scheduler->getAverageSeqlen();
  stat.process_token = scheduler->getNumProcessToken();
  stat.sum_seq = scheduler->getSumSize();
  stat.gen_seq = scheduler->getGenSize();

  if (!config.disagg_system) {
    if (scheduler->hasSumSeq()) {
      stat.latency = time;
      stat.is_mixed = 1;

    } else {
      stat.latency = time;
      stat.is_mixed = 0;
    }
  } else {
    if (!scheduler->hasSumSeq()) {
      stat.latency = time;
      stat.is_mixed = 0;
    }
  }
}

void Cluster::setTimeBreakDown(Stat &stat) {
  // With PP, aggregate stamps from all stages in DP replica 0.
  // Each stage has an independent timeboard; iterate pp=0..P-1 and find_stamp from each.
  int ne_tp_dg = scheduler->model_config.ne_tp_dg;
  int pp_dg = scheduler->model_config.pp_dg;

  // Helper: find_stamp across all PP stages in DP replica 0
  auto find_stamp_all_stages = [&](const std::string& name, std::vector<TimeStamp *>& vec) {
    for (int pp = 0; pp < pp_dg; pp++) {
      int repr_device = pp * ne_tp_dg;
      TimeBoard &tb = get_device(repr_device)->top_module_graph->timeboard;
      tb.find_stamp(name, vec);
    }
  };

  if(scheduler->model_config.qk_nope_head_dim == 0){
    std::vector<TimeStamp *> QKV_gen;    // GPU
    std::vector<TimeStamp *> AttnSum;    // GPU
    std::vector<TimeStamp *> AttnGen;    // PIM or Logic
    std::vector<TimeStamp *> O_proj;     // GPU
    std::vector<TimeStamp *> LMHead;     // lm_head column-parallel linear only
    std::vector<TimeStamp *> FFN;        // PIM or Logic
    std::vector<TimeStamp *> ExpertFFN;  // PIM or Logic
    std::vector<TimeStamp *> Comm;       // PIM or Logic
    std::vector<TimeStamp *> CommInExpertFFN;

    std::vector<TimeStamp *> RoPE;
    std::vector<TimeStamp *> LayerNorm;
    std::vector<TimeStamp *> Residual;

    find_stamp_all_stages("attn_qkv_proj", QKV_gen);
    find_stamp_all_stages("AttentionSum", AttnSum);
    find_stamp_all_stages("AttentionGen", AttnGen);
    find_stamp_all_stages("attn_o_proj", O_proj);
    // lm_head breakdown: col_linear → lm_head, all_gather → communication
    find_stamp_all_stages("lm_head_col_linear", LMHead);
    find_stamp_all_stages("feedforward", FFN);
    find_stamp_all_stages("expertFFN", ExpertFFN);
    find_stamp_all_stages("moe_scatter", CommInExpertFFN);
    find_stamp_all_stages("moe_all_reduce_for_e_tp", CommInExpertFFN);
    find_stamp_all_stages("moe_all_reduce_for_gather", CommInExpertFFN);
    find_stamp_all_stages("moe_gather", CommInExpertFFN);
    find_stamp_all_stages("all_reduce", Comm);
    find_stamp_all_stages("moe_scatter", Comm);
    find_stamp_all_stages("moe_gather", Comm);
    // PP P2P communication
    find_stamp_all_stages("pipeline_send", Comm);
    find_stamp_all_stages("pipeline_recv", Comm);
    // lm_head TP gather → communication
    find_stamp_all_stages("lm_head_all_gather", Comm);
    // embedding TP all-reduce → communication
    find_stamp_all_stages("embedding_all_reduce", Comm);

    find_stamp_all_stages("k_rope", RoPE);
    find_stamp_all_stages("q_rope", RoPE);

    find_stamp_all_stages("input_layer_norm", LayerNorm);
    find_stamp_all_stages("post_attn_layer_norm", LayerNorm);

    find_stamp_all_stages("residual_1", Residual);
    find_stamp_all_stages("residual_2", Residual);

    // Dedicated breakdown columns (also included in communication above):
    std::vector<TimeStamp *> DecodeKVWrite;
    std::vector<TimeStamp *> PipelineRecvWait;
    find_stamp_all_stages("decode_kv_write", DecodeKVWrite);
    find_stamp_all_stages("pipeline_recv", PipelineRecvWait);

    // Energy scaling: each stamp is from one PP stage representative.
    // Scale by devices_per_stage (= dp_degree * ne_tp_dg), not full cluster.
    int dp_degree = num_total_device / (ne_tp_dg * pp_dg);
    int devices_per_stage_energy = dp_degree * ne_tp_dg;

    time_ns qkv_gen = 0;
    time_ns atten_sum = 0;
    time_ns atten_gen = 0;
    time_ns o_proj = 0;
    time_ns ffn = 0;
    time_ns expert_ffn = 0;
    time_ns comm_in_expert_ffn = 0;
    time_ns communication = 0;

    time_ns rope = 0;
    time_ns layernorm = 0;
    time_ns residual = 0;

    energy_nJ FC_DRAM = 0;
    energy_nJ FC_COMP = 0;
    energy_nJ MoE_DRAM = 0;
    energy_nJ MoE_COMP = 0;
    energy_nJ Attn_DRAM = 0;
    energy_nJ Attn_COMP = 0;

    for (auto stamp : QKV_gen) {
      qkv_gen += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.qkv_gen = qkv_gen;

    for (auto stamp : AttnSum) {
      atten_sum += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.atten_sum = atten_sum;

    for (auto stamp : AttnGen) {
      atten_gen += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.atten_gen = atten_gen;

    for (auto stamp : O_proj) {
      o_proj += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.o_proj = o_proj;

    time_ns lm_head_time = 0;
    for (auto stamp : LMHead) {
      lm_head_time += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.lm_head = lm_head_time;

    for (auto stamp : FFN) {
      ffn += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy;
    }
    stat.ffn = ffn;

    for (auto stamp : ExpertFFN) {
      expert_ffn += stamp->get_duration();
    }

    // expertFFN may have different energy by device
    for(int device_id = 0; device_id < num_total_device; device_id ++){
      TimeBoard &timeboard_temp = get_device(device_id)->top_module_graph->timeboard;
      std::vector<TimeStamp *> ExpertFFN_temp;  // PIM or Logic
      timeboard_temp.find_stamp("expertFFN", ExpertFFN_temp);
      for (auto stamp : ExpertFFN_temp) {
        MoE_DRAM += stamp->getDramEnergy();
        MoE_COMP += stamp->getCompEnergy();
      }
    }

    for (auto stamp : CommInExpertFFN) {
      comm_in_expert_ffn += stamp->get_duration();
    }

    stat.expert_ffn = expert_ffn - comm_in_expert_ffn;

    for (auto stamp : Comm) {
      communication += stamp->get_duration();
    }
    stat.communication = communication;

    for (auto stamp : RoPE) {
      rope += stamp->get_duration();
    }
    stat.rope = rope;

    for (auto stamp : LayerNorm) {
      layernorm += stamp->get_duration();
    }
    stat.layernorm = layernorm;

    for (auto stamp : Residual) {
      residual += stamp->get_duration();
    }
    stat.residual = residual;

    // Previously missing breakdown items (time-only, no energy tracking in v1)
    time_ns dkv_time = 0;
    for (auto stamp : DecodeKVWrite) dkv_time += stamp->get_duration();
    stat.decode_kv_write = dkv_time;

    time_ns recv_wait = 0;
    for (auto stamp : PipelineRecvWait) recv_wait += stamp->get_duration();
    stat.pipeline_recv_wait = recv_wait;

    stat.FC_DRAM_energy = FC_DRAM;
    stat.FC_COMP_energy = FC_COMP;
    stat.Attn_DRAM_energy = Attn_DRAM;
    stat.Attn_COMP_energy = Attn_COMP;
    stat.MoE_DRAM_energy = MoE_DRAM;
    stat.MoE_COMP_energy = MoE_COMP;
    stat.isOOM = out_of_memory;

    double opb = 0;
    for (auto stamp : AttnSum) {
      opb += stamp->getOpb();
    }

    if (AttnSum.size()) {
      opb /= AttnSum.size();
    }
    stat.sum_attention_opb = opb;
  }
  else{ // if Use MLA
    std::vector<TimeStamp *> Decoders;    
    std::vector<TimeStamp *> Q_down;    
    std::vector<TimeStamp *> KV_down;    
    std::vector<TimeStamp *> KR_proj;    

    std::vector<TimeStamp *> RoPE;
    std::vector<TimeStamp *> LayerNorm;
    std::vector<TimeStamp *> Residual;

    std::vector<TimeStamp *> Q_up;    
    std::vector<TimeStamp *> QR_proj;    
    std::vector<TimeStamp *> KV_up;

    // for Absorb Impl //
    std::vector<TimeStamp *> tr_K_up;
    std::vector<TimeStamp *> V_up;
    
    std::vector<TimeStamp *> AttnSum;    
    std::vector<TimeStamp *> AttnGen;    

    std::vector<TimeStamp *> O_proj;
    std::vector<TimeStamp *> LMHead;     // lm_head column-parallel linear only

    std::vector<TimeStamp *> FFN;
    std::vector<TimeStamp *> ExpertFFN;
    std::vector<TimeStamp *> Comm;
    std::vector<TimeStamp *> CommInExpertFFN;
    std::vector<TimeStamp *> Test;

    find_stamp_all_stages("attn_q_down_proj", Q_down);
    find_stamp_all_stages("attn_kv_down_proj", KV_down);
    find_stamp_all_stages("attn_kr_proj", KR_proj);

    find_stamp_all_stages("attn_q_up_proj", Q_up);
    find_stamp_all_stages("attn_qr_proj", QR_proj);
    find_stamp_all_stages("attn_kv_up_proj", KV_up);

    // for MLA absorb //
    find_stamp_all_stages("attn_tr_k_up_proj", tr_K_up);
    find_stamp_all_stages("attn_v_up_proj", V_up);

    find_stamp_all_stages("AttentionSum", AttnSum);
    find_stamp_all_stages("AttentionGen", AttnGen);

    find_stamp_all_stages("attn_o_proj", O_proj);
    // lm_head breakdown: col_linear → lm_head, all_gather → communication
    find_stamp_all_stages("lm_head_col_linear", LMHead);

    find_stamp_all_stages("feedforward", FFN);
    find_stamp_all_stages("expertFFN", ExpertFFN);
    find_stamp_all_stages("moe_scatter", CommInExpertFFN);
    find_stamp_all_stages("moe_all_reduce_for_e_tp", CommInExpertFFN);
    find_stamp_all_stages("moe_all_reduce_for_gather", CommInExpertFFN);
    find_stamp_all_stages("moe_gather", CommInExpertFFN);
    find_stamp_all_stages("all_reduce", Comm);
    find_stamp_all_stages("moe_scatter", Comm);
    find_stamp_all_stages("moe_gather", Comm);
    // PP P2P communication
    find_stamp_all_stages("pipeline_send", Comm);
    find_stamp_all_stages("pipeline_recv", Comm);
    // lm_head TP gather → communication
    find_stamp_all_stages("lm_head_all_gather", Comm);
    // embedding TP all-reduce → communication
    find_stamp_all_stages("embedding_all_reduce", Comm);

    find_stamp_all_stages("k_rope", RoPE);
    find_stamp_all_stages("q_rope", RoPE);

    find_stamp_all_stages("input_layer_norm", LayerNorm);
    find_stamp_all_stages("latent_q_layer_norm", LayerNorm);
    find_stamp_all_stages("latent_kv_layer_norm", LayerNorm);
    find_stamp_all_stages("post_attn_layer_norm", LayerNorm);

    find_stamp_all_stages("residual_1", Residual);
    find_stamp_all_stages("residual_2", Residual);

    // Dedicated breakdown columns (also included in communication above):
    std::vector<TimeStamp *> DecodeKVWrite;
    std::vector<TimeStamp *> PipelineRecvWait;
    find_stamp_all_stages("decode_kv_write", DecodeKVWrite);
    find_stamp_all_stages("pipeline_recv", PipelineRecvWait);

    int dp_degree_mla = num_total_device / (ne_tp_dg * pp_dg);
    int devices_per_stage_energy_mla = dp_degree_mla * ne_tp_dg;

    find_stamp_all_stages("decoder_", Decoders);

    time_ns q_down_proj = 0;
    time_ns kv_down_proj = 0;
    time_ns kr_proj = 0;

    time_ns q_up_proj = 0;
    time_ns qr_proj = 0;
    time_ns kv_up_proj = 0;

    // for MLA absorb //
    time_ns tr_k_up_proj = 0;
    time_ns v_up_proj = 0;
    // 

    time_ns atten_sum = 0;
    time_ns atten_gen = 0;
    time_ns o_proj = 0;
    time_ns ffn = 0;
    time_ns expert_ffn = 0;
    time_ns comm_in_expert_ffn = 0;
    time_ns communication = 0;
    
    time_ns rope = 0;
    time_ns layernorm = 0;
    time_ns residual = 0;

    energy_nJ FC_DRAM = 0;
    energy_nJ FC_COMP = 0;
    energy_nJ MoE_DRAM = 0;
    energy_nJ MoE_COMP = 0;
    energy_nJ Attn_DRAM = 0;
    energy_nJ Attn_COMP = 0;

    for (auto stamp : Q_down){
      q_down_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.q_down_proj = q_down_proj;

    for (auto stamp : KV_down){
      kv_down_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.kv_down_proj = kv_down_proj;

    for (auto stamp : KR_proj){
      kr_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.kr_proj = kr_proj;

    for (auto stamp : Q_up){
      q_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.q_up_proj = q_up_proj;

    for (auto stamp : QR_proj){
      qr_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.qr_proj = qr_proj;

    for (auto stamp : KV_up){
      kv_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.kv_up_proj = kv_up_proj;

    for (auto stamp : tr_K_up){
      tr_k_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.tr_k_up_proj = tr_k_up_proj;

    for (auto stamp : V_up){
      v_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.v_up_proj = v_up_proj;

    for (auto stamp : AttnSum) {
      atten_sum += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.atten_sum = atten_sum;

    for (auto stamp : AttnGen) {
      atten_gen += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      Attn_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.atten_gen = atten_gen;

    for (auto stamp : O_proj) {
      o_proj += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.o_proj = o_proj;

    time_ns lm_head_time = 0;
    for (auto stamp : LMHead) {
      lm_head_time += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.lm_head = lm_head_time;

    for (auto stamp : FFN) {
      ffn += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * devices_per_stage_energy_mla;
      FC_COMP += stamp->getCompEnergy() * devices_per_stage_energy_mla;
    }
    stat.ffn = ffn;

    for (auto stamp : ExpertFFN) {
      expert_ffn += stamp->get_duration();
    }

    // expertFFN may have different energy by device
    for(int device_id = 0; device_id < num_total_device; device_id ++){
      TimeBoard &timeboard_temp = get_device(device_id)->top_module_graph->timeboard;
      std::vector<TimeStamp *> ExpertFFN_temp;  // PIM or Logic
      timeboard_temp.find_stamp("expertFFN", ExpertFFN_temp);
      for (auto stamp : ExpertFFN_temp) {
        MoE_DRAM += stamp->getDramEnergy();
        MoE_COMP += stamp->getCompEnergy();
      }
    }

    for (auto stamp : CommInExpertFFN) {
      comm_in_expert_ffn += stamp->get_duration();
    }

    stat.expert_ffn = expert_ffn - comm_in_expert_ffn;

    for (auto stamp : Comm) {
      communication += stamp->get_duration();
    }
    stat.communication = communication;

    for (auto stamp : RoPE) {
      rope += stamp->get_duration();
    }
    stat.rope = rope;

    for (auto stamp : LayerNorm) {
      layernorm += stamp->get_duration();
    }
    stat.layernorm = layernorm;

    for (auto stamp : Residual) {
      residual += stamp->get_duration();
    }
    stat.residual = residual;

    // Previously missing breakdown items (time-only, no energy tracking in v1)
    time_ns dkv_time = 0;
    for (auto stamp : DecodeKVWrite) dkv_time += stamp->get_duration();
    stat.decode_kv_write = dkv_time;

    time_ns recv_wait = 0;
    for (auto stamp : PipelineRecvWait) recv_wait += stamp->get_duration();
    stat.pipeline_recv_wait = recv_wait;

    stat.FC_DRAM_energy = FC_DRAM;
    stat.FC_COMP_energy = FC_COMP;
    stat.Attn_DRAM_energy = Attn_DRAM;
    stat.Attn_COMP_energy = Attn_COMP;
    stat.MoE_DRAM_energy = MoE_DRAM;
    stat.MoE_COMP_energy = MoE_COMP;
    stat.isOOM = out_of_memory;

    double opb = 0;
    for (auto stamp : AttnSum) {
      opb += stamp->getOpb();
    }

    if (AttnSum.size()) {
      opb /= AttnSum.size();
    }
    stat.sum_attention_opb = opb;
  }

  // dowon revised
  // Within one round, stamps from all microbatches accumulate in the timeboard
  // because restartGraphOnly() preserves the timeboard between microbatches.
  // The timeboard is reset at round boundaries. The breakdown is normalized by pp_dg.
  if (pp_dg > 1) {
    stat.qkv_gen /= pp_dg;
    stat.q_down_proj /= pp_dg;
    stat.kv_down_proj /= pp_dg;
    stat.kr_proj /= pp_dg;
    stat.q_up_proj /= pp_dg;
    stat.qr_proj /= pp_dg;
    stat.kv_up_proj /= pp_dg;
    stat.tr_k_up_proj /= pp_dg;
    stat.v_up_proj /= pp_dg;
    stat.atten_sum /= pp_dg;
    stat.atten_gen /= pp_dg;
    stat.o_proj /= pp_dg;
    stat.lm_head /= pp_dg;
    stat.ffn /= pp_dg;
    stat.expert_ffn /= pp_dg;
    stat.communication /= pp_dg;
    stat.decode_kv_write /= pp_dg;
    stat.pipeline_recv_wait /= pp_dg;
    stat.rope /= pp_dg;
    stat.layernorm /= pp_dg;
    stat.residual /= pp_dg;
  }
}

void Cluster::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  setPerformExecution(true);
  restartModuleGraph();
  while (check_module_graph_remain()) {
    for (Node::Ptr _node : node) {
      _node->run(sequences_metadata_list);
    }
  }
}

// dowon revised
// Between microbatches within one round: only reset graph iterator + tensor flags.
// device_time, energy, timeboard all preserved so microbatch stamps accumulate.
void Cluster::restartGraphOnly() {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    get_device(device_rank)->restartGraph();
    // Do NOT reset_status — preserve device_time and energy
    // Do NOT reset_timeboard — stamps accumulate across microbatches
  }
}

// dowon revised
// Between rounds: reset graph/timeboard/energy, preserve device_time.
// Unlike restartForNextIteration(), does NOT clamp stage 0 to last stage end.
// Per-sequence autoregressive dependency is handled by next_decode_ready_time
// in launchMicrobatch.
void Cluster::restartForNextRoundNoClamp() {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr dev = get_device(device_rank);
    time_ns saved_device_time = dev->status.device_time;
    time_ns saved_high_time = dev->status.high_time;
    time_ns saved_low_time = dev->status.low_time;
    dev->restartGraph();
    dev->reset_status();
    dev->reset_timeboard();
    dev->status.device_time = saved_device_time;
    dev->status.high_time = saved_high_time;
    dev->status.low_time = saved_low_time;
  }
}

// dowon revised
// Run one decode microbatch through the full module graph.
// microbatch_metas[dp] = BatchedSequence for each DP replica in this microbatch.
// Returns per-DP last stage finish times.
std::vector<time_ns> Cluster::runOneMicrobatch(
    std::vector<BatchedSequence::Ptr>& microbatch_metas) {
  int ne_tp = scheduler->model_config.ne_tp_dg;
  int pp = scheduler->model_config.pp_dg;
  int dp = microbatch_metas.size();

  // Swap running_queue so GateUpdate::aggregate_expert sees microbatch metadata
  auto original_queue = scheduler->running_queue;
  scheduler->running_queue = microbatch_metas;

  while (check_module_graph_remain()) {
    for (Node::Ptr _node : node)
      _node->run(microbatch_metas);
  }

  scheduler->running_queue = original_queue;

  // Per-DP last stage finish time (TP group max within each DP replica)
  std::vector<time_ns> finishes(dp);
  for (int d = 0; d < dp; d++) {
    int last_stage_base = d * (pp * ne_tp) + (pp - 1) * ne_tp;
    time_ns f = 0;
    for (int t = 0; t < ne_tp; t++)
      f = std::max(f, get_device(last_stage_base + t)->status.device_time);
    finishes[d] = f;
  }
  return finishes;
}

// dowon revised
// =============================================================================
// PP decode pipeline main loop (persistent microbatches).
// Eliminates the global round barrier that inflates token_interval.
//
// Key difference from the old transient-microbatch path:
// - Microbatches are persistent (created once, sequences stay assigned)
// - Per-sequence next_decode_ready_time controls when it can launch (not a global barrier)
// - No stage 0 clamp between rounds
// - Same sequence's next token starts when: next_decode_ready_time <= stage0_time
//
// Scope: decode_mode && pp_dg > 1.
// =============================================================================
std::vector<Stat> Cluster::runIterationPipelineDecode(
    int iter, std::ofstream& csv) {
  constexpr int kCSVExportInterval = 25;

  time_ns total_time = 0;
  std::vector<Stat> stat_list;
  auto start = std::chrono::high_resolution_clock::now();

  int pp_dg = scheduler->model_config.pp_dg;
  int ne_tp = scheduler->model_config.ne_tp_dg;
  int dp = num_total_device / (ne_tp * pp_dg);

  // dowon revised
  // Initialize persistent microbatches from all DP replicas in running_queue
  scheduler->initializeMicrobatches();

  // dowon revised
  // Single trace steady state: seed output progress in microbatches
  if (config.single_trace_steady_state) {
    scheduler->seedSteadyStateMicrobatches();
    seeded_remaining = 0;
    for (int s = 0; s < pp_dg; s++)
      for (int d = 0; d < dp; d++)
        for (auto& seq : scheduler->microbatches[s][d]->sequence)
          if (seq->turnover_seed) seeded_remaining++;
    std::cerr << "Single-trace steady state: seeded " << seeded_remaining
              << " sequences across " << pp_dg << " slots" << std::endl;
  }

  for (int round = 0; round < iter; round++) {
    // CSV export
    if (round % kCSVExportInterval == kCSVExportInterval - 1) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - start);
      start = std::chrono::high_resolution_clock::now();
      exportToCSVNoEnergy(csv, stat_list);
    }

    // Record round base time (for wall-clock delta)
    time_ns round_base = 0;
    for (int d = 0; d < num_total_device; d++)
      round_base = std::max(round_base, get_device(d)->status.device_time);

    // First round: full reset. Subsequent: graph/timeboard/energy reset, keep time.
    setPerformExecution(true);
    if (round == 0)
      restartModuleGraph();
    else
      restartForNextRoundNoClamp();

    // dowon revised
    // Execute each microbatch in order within this round.
    // Between microbatches: restartGraphOnly (reset graph, keep time).
    // Stage overlap: PipelineRecv syncs naturally via device_time.
    std::vector<Sequence::Ptr> round_milestones;
    int total_launched_tokens = 0;

    for (int s = 0; s < pp_dg; s++) {
      if (s > 0) restartGraphOnly();

      // Per-DP launch times: stage 0 TP group max device_time for each DP replica
      std::vector<time_ns> stage0_launch_times(dp);
      for (int d = 0; d < dp; d++) {
        int stage0_base = d * (pp_dg * ne_tp);
        time_ns t = 0;
        for (int tp = 0; tp < ne_tp; tp++)
          t = std::max(t, get_device(stage0_base + tp)->status.device_time);
        stage0_launch_times[d] = t;
      }

      // dowon revised
      // Per-DP: if any sequence in this microbatch is not yet ready, advance stage 0
      // to the latest next_decode_ready_time among microbatch members. A persistent
      // microbatch models a microbatch lane, so it launches together rather than
      // partially launching only the earliest-ready subset.
      for (int d = 0; d < dp; d++) {
        time_ns latest_ready_time = 0;
        bool has_ready = false;
        for (auto& seq : scheduler->microbatches[s][d]->sequence) {
          if (seq->current_len >= seq->total_len) continue;
          if (seq->in_flight) continue;
          if (!has_ready || seq->next_decode_ready_time > latest_ready_time) {
            latest_ready_time = seq->next_decode_ready_time;
            has_ready = true;
          }
        }

        if (has_ready && latest_ready_time > stage0_launch_times[d]) {
          stage0_launch_times[d] = latest_ready_time;
          int stage0_base = d * (pp_dg * ne_tp);
          for (int tp = 0; tp < ne_tp; tp++) {
            Device::Ptr dev = get_device(stage0_base + tp);
            dev->status.device_time = std::max(dev->status.device_time, latest_ready_time);
            dev->status.high_time = dev->status.device_time;
            dev->status.low_time = dev->status.device_time;
          }
        }
      }

      // Launch across all DP replicas
      int launched_tokens = scheduler->launchMicrobatch(s, stage0_launch_times);
      total_launched_tokens += launched_tokens;
      if (launched_tokens == 0) continue;

      // Run this microbatch through the full graph
      auto finish_per_dp = runOneMicrobatch(scheduler->microbatches[s]);

      // Complete: update sequences with per-DP absolute finish times
      auto milestones = scheduler->completeMicrobatch(s, finish_per_dp);
      round_milestones.insert(round_milestones.end(),
          milestones.begin(), milestones.end());

      // Track seeded_remaining: count turnovers in this microbatch completion
      if (config.single_trace_steady_state) {
        for (auto& seq : milestones) {
          if (seq->turnover_seed && seq->current_len >= seq->total_len) {
            seeded_remaining--;
          }
        }
      }
    }

    // Round wall-clock
    time_ns round_end = 0;
    for (int d = 0; d < num_total_device; d++)
      round_end = std::max(round_end, get_device(d)->status.device_time);
    time_ns time = round_end - round_base;

    // Skip round if nothing was launched (no active sequences)
    if (total_launched_tokens == 0) {
      time = 20 * 1000 * 1000;
      continue;
    }

    total_time += time;

    // Stat collection (same structure as runIterationMixed)
    Stat stat;
    stat.iter_info = 1;
    stat.type = "t2t";
    stat.time = total_time;
    scheduler->total_time = total_time;

    std::vector<energy_nJ> total_energy = getTotalEnergy();
    stat.act_energy = total_energy[0];
    stat.read_energy = total_energy[1];
    stat.write_energy = total_energy[2];
    stat.all_act_energy = total_energy[3];
    stat.all_read_energy = total_energy[4];
    stat.all_write_energy = total_energy[5];
    stat.mac_energy = total_energy[6];
    stat.total_energy = total_energy[7];
    stat.seq_queue_size = scheduler->sequence_queue.size();

    setStat(stat);
    setTimeBreakDown(stat);

    // dowon revised
    // For PP decode microbatch mode, stat.latency = round wall-clock time.
    // token_interval is computed per-sequence in addLatency using
    // the absolute end_token_time and second_token_time set by completeMicrobatch.
    stat.latency = time;

    // Single trace steady state: TPS tracking only. Early termination is
    // intentionally disabled so single_trace_steady_state respects iter.
    if (config.single_trace_steady_state) {
      // Round 0 = warmup (includes initial KV write) → exclude from TPS
      if (round == 0) {
        tps_started = false;
      } else if (!tps_started) {
        tps_started = true;
      }

      if (tps_started && !tps_finished) {
        cumulative_generated_tokens += total_launched_tokens;
        cumulative_elapsed_time += time;
      }

      // Fill stat fields
      stat.generated_tokens = tps_started ? total_launched_tokens : 0;
      stat.cumulative_generated_tokens = cumulative_generated_tokens;
      if (cumulative_elapsed_time > 0 && tps_started) {
        stat.tps = (double)cumulative_generated_tokens * 1e9 / cumulative_elapsed_time;
        stat.tps_per_gpu = stat.tps / num_total_device;
      }
      stat.seeded_remaining = seeded_remaining;
      stat.first_departure_seen = first_seeded_departure_seen ? 1 : 0;
      stat.post_departure_steps = post_departure_decode_steps;
    }

    finalizeThreeYearPEC(stat);
    stat_list.push_back(stat);
    addLatency(stat_list, round_milestones, total_time);

    // dowon revised
    // Refill: add new sequences from queue to microbatches using the original
    // running_queue admission policy.
    scheduler->fillSequenceQueue(time, total_time);
    scheduler->fillRunningQueue();
    scheduler->refillMicrobatches();
  }

  return stat_list;
}

void Cluster::setPerformExecution(bool perform) {
  for (Node::Ptr _node : node) {
    _node->setPerformExecution(perform);
  }
};

void Cluster::set(SystemConfig config) {
  CreateNode(config);
  module_map.resize(num_total_device);
}

void Cluster::CreateNode(SystemConfig config) {
  for (int node_rank = 0; node_rank < config.num_node; node_rank++) {
    node.push_back(Node::Create(config, node_rank, getptr()));
  }
}

};  // namespace llm_system
