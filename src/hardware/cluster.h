#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/hardware_config.h"
#include "hardware/node.h"
#include "hardware/stat.h"
#include "module/module.h"
#include "scheduler/scheduler.h"
#include "dram/power.h"

namespace llm_system {

class Cluster : public std::enable_shared_from_this<Cluster> {
  friend class Device;

 public:
  using Ptr = std::shared_ptr<Cluster>;
  [[nodiscard]] static Ptr Create(SystemConfig config,
                                  Scheduler::Ptr scheduler) {
    Ptr cluster = Ptr(new Cluster(config, scheduler));
    cluster->set(config);
    cluster->initializeDRAM((int)(ProcessorType::GPU), gpuEnergy);
    cluster->initializeDRAM((int)(ProcessorType::LOGIC), logicEnergy);
    cluster->initializeDRAM((int)(ProcessorType::PIM), pimEnergy);
    return cluster;
  }

  hw_metric cluster_ict_latency;
  hw_metric cluster_ict_bandwidth;

  std::vector<Node::Ptr> node;

  Ptr getptr() { return shared_from_this(); }

  Cluster(Cluster &&) = default;
  Cluster &operator=(Cluster &&) = default;

  SystemConfig config;
  int num_device;
  int num_node;
  int num_total_device;
  bool out_of_memory = false;

  Device::Ptr get_device(int device_total_rank);

  void set_dependency();
  void add_module(int device_rank, std::string name, Module::Ptr module);

  void set_dependency_tensor(std::vector<Tensor::Ptr> &list, Tensor::Ptr tensor,
                             const std::vector<int> &device_list);

  // Look up a specific tensor by device rank, module name, and tensor name.
  // Used by PipelineRecv to find PipelineSend's output on a remote device.
  Tensor::Ptr find_tensor(int device_rank, const std::string& module_map_name,
                          const std::string& tensor_name);

  std::vector<Stat> runIteration(int iter, std::string file_name = "stat");

  std::vector<Stat> runIterationMixed(int iter, std::ofstream &csv);
  std::vector<Stat> runIterationSumGenSplit(int iter, std::ofstream &csv);

  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);
  void restartModuleGraph();

  // dowon revised
  // Between microbatches within one round: reset only graph iterator/tensor flags.
  // Preserves device_time, energy, timeboard so microbatch stamps accumulate.
  void restartGraphOnly();

  // dowon revised
  // Between rounds: reset graph/timeboard/energy, preserve device_time.
  // Unlike restartForNextIteration(), no stage 0 clamp; stage 0 continues from
  // its own device_time. Per-sequence autoregressive dependency comes from
  // next_decode_ready_time.
  void restartForNextRoundNoClamp();

  // dowon revised
  // Run one decode microbatch through the full graph.
  // microbatch_metas[dp] = microbatch's BatchedSequence per DP replica.
  // Returns per-DP last stage finish times.
  std::vector<time_ns> runOneMicrobatch(
      std::vector<BatchedSequence::Ptr>& microbatch_metas);

  // dowon revised
  // PP decode path using persistent microbatches.
  // Used when decode_mode && pp_dg > 1.
  std::vector<Stat> runIterationPipelineDecode(int iter, std::ofstream& csv);

  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  void setPerformExecution(bool perform);

  void updateTimestamp();
  void calibrateMemPoolLoadTime(Stat &stat);

  std::vector<std::map<std::string, Module::Ptr>> module_map;

  bool checkMemorySize();
  bool checkHeteroMemorySize();
  std::vector<energy_nJ> getTotalEnergy();

  void setTimeBreakDown(Stat &stat);
  void setStat(Stat &stat);

  void addLatency(std::vector<Stat> &stat_list,
                  const std::vector<Sequence::Ptr> &seq_list, time_ns time);

  void exportGantt(std::string gantt_file_path);

  std::map<CacheKey, ExecStatus> execution_time_cache;

  // Single trace steady state TPS tracking (runtime state)
  bool tps_started = false;
  bool tps_finished = false;
  long long cumulative_generated_tokens = 0;
  time_ns cumulative_elapsed_time = 0;
  int seeded_remaining = 0;

  // Early termination: stop after first seeded departure + 1 decode step
  bool first_seeded_departure_seen = false;
  bool run_one_more_decode_step = false;
  int post_departure_decode_steps = 0;

 private:
  bool check_module_graph_remain();
  void exportToCSV(std::ofstream &csv, std::vector<Stat> &stat_list);
  // CSV export without energy columns. To restore energy, replace
  // exportToCSVNoEnergy() calls with exportToCSV() and use the full header.
  void exportToCSVNoEnergy(std::ofstream &csv, std::vector<Stat> &stat_list);
  void set(SystemConfig config);
  Cluster(SystemConfig config, Scheduler::Ptr scheduler);
  Cluster() = default;

  // PEC baseline state (three_year_pec)
  bool pec_baseline_initialized = false;
  time_ns pec_start_time = 0;
  std::vector<long long> pec_baseline_kv_write_bytes;  // per device

  // PEC helpers
  void resetPECState();
  void finalizeThreeYearPEC(Stat& stat);
  double computeThreeYearPEC(time_ns current_time, const std::string& type);

  Executor executor;

  Scheduler::Ptr scheduler;

  void CreateNode(SystemConfig config);
};

}  // namespace llm_system