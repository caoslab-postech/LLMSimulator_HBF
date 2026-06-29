#pragma once
#include <string>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"

namespace llm_system {

struct PIMHWConfig {
  ProcessorType type = ProcessorType::GPU;

  int bandwidth_x = 0;
};

class SystemConfig {

  public:
    // default initilizing with H100 config
    SystemConfig(std::string gpu_gen ="H100",
                 int num_node = 1, int num_device = 2, 
                 hw_metric node_ict_latency = 0.5 * 1000,
                 hw_metric node_ict_bandwidth = 400.0 * 1000 * 1000 * 1000,
                 hw_metric device_ict_latency = 3.0 * 1000,
                 hw_metric device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000, 
                 hw_metric compute_peak_flops = 989.4 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_read_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_write_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric intermediate_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric intermediate_buffer = 0.0,
                 bool intermediate_overlap = false,
                 hw_metric memory_capacity = 80.0 * 1024 * 1024 * 1024,
                 int logic_x = 4,
                 hw_metric logic_op_b = 8,                 
                 int pim_x = 16,
                 hw_metric pim_op_b = 1,
                 std::vector<ProcessorType> processor_type = {},
                 bool parallel_execution = false,
                 bool hetero_subbatch = false,
                 ProcessorType high_processor_type = ProcessorType::GPU,
                 ProcessorType low_processor_type = ProcessorType::LOGIC,
                 bool communication_hiding = false,            
                 bool disagg_system = false,
                 bool use_low_unit_moe_only = false,
                 bool use_ramulator = false,
                 bool exit_out_of_memory = true,
                 bool mem_cap_limit = false,               
                 bool use_flash_mla = true,
                 bool use_flash_attention = true,
                 bool reuse_kv_cache = true,
                 hw_metric kv_cache_reuse_rate = 0.5,
                 bool prefill_mode = false,
                 bool decode_mode = false,
                 bool use_inject_rate = false,
                 int request_per_second = 10,
                 int num_cube = 5,
                 int num_logic_cube = 5
                )
      : gpu_gen(gpu_gen),
        num_node(num_node),
        num_device(num_device),
        node_ict_latency(node_ict_latency),
        node_ict_bandwidth(node_ict_bandwidth),
        device_ict_latency(device_ict_latency),
        device_ict_bandwidth(device_ict_bandwidth),
        compute_peak_flops(compute_peak_flops),
        memory_bandwidth(memory_bandwidth),
        memory_read_bandwidth(memory_read_bandwidth),
        memory_write_bandwidth(memory_write_bandwidth),
        intermediate_bandwidth(intermediate_bandwidth),
        intermediate_buffer(intermediate_buffer),
        intermediate_overlap(intermediate_overlap),
        memory_capacity(memory_capacity),
        logic_x(logic_x),
        logic_op_b(logic_op_b),
        pim_x(pim_x),
        pim_op_b(pim_op_b),
        processor_type(processor_type),
        parallel_execution(parallel_execution),
        hetero_subbatch(hetero_subbatch),
        high_processor_type(high_processor_type),
        low_processor_type(low_processor_type),
        communication_hiding(communication_hiding),
        disagg_system(disagg_system),
        use_low_unit_moe_only(use_low_unit_moe_only),
        use_ramulator(use_ramulator),
        exit_out_of_memory(exit_out_of_memory),
        mem_cap_limit(mem_cap_limit),
        use_flash_mla(use_flash_mla),
        use_flash_attention(use_flash_attention),
        reuse_kv_cache(reuse_kv_cache),
        kv_cache_reuse_rate(kv_cache_reuse_rate),
        prefill_mode(prefill_mode),
        decode_mode(decode_mode),
        use_inject_rate(use_inject_rate),
        request_per_second(request_per_second),
        num_cube(num_cube),
        num_logic_cube(num_logic_cube){
          logic_memory_bandwidth = (memory_bandwidth) / 2.0 * logic_x;
          pim_memory_bandwidth = (memory_bandwidth) / 2.0 * pim_x;
        };

    SystemConfig& operator=(const SystemConfig& rhs) = default;

  std::string gpu_gen;

  // Device number
  int num_node;
  int num_device;

  // Cluster specification
  hw_metric node_ict_latency;   // ns
  hw_metric node_ict_bandwidth; // B/s

  // Node specification
  hw_metric device_ict_latency;    // ns, 
  hw_metric device_ict_bandwidth;  // B/s

  // Device specification
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  hw_metric memory_read_bandwidth;
  hw_metric memory_write_bandwidth;
  hw_metric intermediate_bandwidth;
  hw_metric intermediate_buffer;

  bool intermediate_overlap;

  hw_metric memory_capacity;

  // Logic specification
  int logic_x;
  hw_metric logic_memory_bandwidth = memory_bandwidth * logic_x;
  hw_metric logic_op_b;

  // PIM specifiaction
  int pim_x;
  hw_metric pim_memory_bandwidth = memory_bandwidth * pim_x;
  hw_metric pim_op_b;

  std::vector<ProcessorType> processor_type = {};

  bool parallel_execution = false;
  bool hetero_subbatch = false;
  ProcessorType high_processor_type = ProcessorType::GPU;
  ProcessorType low_processor_type = ProcessorType::LOGIC;

  bool communication_hiding = false;

  bool disagg_system = true;
  bool use_low_unit_moe_only = false;
  bool use_ramulator = false;
  
  bool exit_out_of_memory = false;
  bool mem_cap_limit = false;
  bool ignore_activation_buffer_oom = false;

  bool use_flash_mla = true; 
  bool use_flash_attention = true; 
  bool reuse_kv_cache = true;
  hw_metric kv_cache_reuse_rate; 
  // this rate includes, 
  // 1) how long does prompt share tokens with cached KV 
  // 2) does prompt share tokens with cached KV
  // because we select rate between [0, kv_cache_reuse_rate * 2), kv_cache_reuse_rate must be max 0.5

  bool prefill_mode = false; 
  bool decode_mode = false;

  bool use_inject_rate = false;  // injection random number of sequence
  int request_per_second;

  // Single trace steady state: seed batch with uniform output progress,
  // measure accumulated TPS excluding initial warmup round.
  bool single_trace_steady_state = false;

  int num_cube; //8: for HBM3E (B100), 5 for HBM3 (H100)
  int num_logic_cube;
  // Device
};


static SystemConfig A100 = SystemConfig(
                 "A100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 https://www.fs.com/products/161048.html?attribute=106827&id=3941024
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 3.0 * 1000,                        // device_ict_latency, nvlink 3
                 150.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 312.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.039 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 2.039 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 2.039 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 2.039 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system 
                 false,                             // use_low_unit_moe_only
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig H100 = SystemConfig(
                 "H100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth, nvlink 4
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 3.352 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 3.352 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 3.352 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 3.352 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig H200 = SystemConfig(
                 "H200",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth, nvlink 4
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 4.8 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 4.8 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 4.8 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 4.8 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 141.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

static SystemConfig B100 = SystemConfig(
                  "B100",                            // gpu gen
                  1,                                 // num_node 
                  2,                                 // num_device
                  130.0,                             // node_ict_latency, connectx-7 
                  50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                  0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                  900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                  1750.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                  8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                  8.000 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                  8.000 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                  8.000 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                  0.0,                               // intermediate_buffer
                  false,                             // intermediate_overlap
                  192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                  4,                                 // logic_x 
                  8,                                 // logic_op_b                 
                  16,                                // pim_x
                  1,                                 // pim_op_b
                  {},                                // processor_type
                  false,                             // parallel_execution
                  false,                             // hetero_subbatch
                  ProcessorType::GPU,                // high_processor_type
                  ProcessorType::LOGIC,              // low_processor_type
                  false,                             // communication_hiding
                  false,                             // disagg_system
                  false,                             // use_low_unit_moe_only 
                  false,                             // use_ramulator
                  true,                              // exit_out_of_memory
                  false,                             // mem_cap_limit
                  true,                              // use_flash_mla
                  true,                              // use_flash_attention
                  false,                             // reuse_kv_cache
                  0.0,                               // kv_cache_reuse_rate
                  false,                             // prefill_mode
                  false,                             // decode_mode
                  false,                             // use_inject_rate
                  10,                                // request_per_second
                  8,                                 // num_cube
                  8                                 // int num_logic_cube
                  );


                  
static SystemConfig B200 = SystemConfig(
                 "B200",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 2250.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 8.000 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 8.000 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 8.000 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                 // int num_logic_cube
                 );
static SystemConfig R100 = SystemConfig(
                 "R100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 22.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 22.000 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 22.000 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 22.000 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 288.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );


static SystemConfig HBM4 = SystemConfig(
                 "HBM4",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 12.8 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000, // memory_read_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000, // memory_write_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000, // intermediate_bandwidth
                 0.0,                               // intermediate_buffer
                 false,                             // intermediate_overlap
                 288.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );

// HBF GPUs
// HBF-Cons: tR=3us, 16 dies/plane, 

// HBF_Cons_5:one HBM stack for activation buffer, compare with H200
static SystemConfig HBF_Cons_5 = SystemConfig(
                 "HBF_Cons_5",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 1.75  * 1000 * 1000 * 1000 * 1000,   // memory_bandwidth
                 1.75  * 1000 * 1000 * 1000 * 1000,   // memory_read_bandwidth
                 52.43 * 1000 * 1000 * 1000,           // memory_write_bandwidth
                 0.8 * 1000 * 1000 * 1000 * 1000,   // intermediate_bandwidth, use one HBM3e stack of H200
                 24.0 * 1 * 1024 * 1024 * 1024,     // intermediate_buffer HBM3e stack capacity
                 true,                              // intermediate_overlap
                 512.0 * 5 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF_Cons_6:sufficient SRAM for activation buffer, compare with H200
static SystemConfig HBF_Cons_6 = SystemConfig(
                 "HBF_Cons_6",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.10 * 1000 * 1000 * 1000 * 1000,  // memory_bandwidth
                 1.92 * 1000 * 1000 * 1000 * 1000,  // memory_read_bandwidth
                 62.91 * 1000 * 1000 * 1000,         // memory_write_bandwidth
                 2.10 * 1000 * 1000 * 1000 * 1000,  // intermediate_bandwidth
                 38.0 * 6 * 1024 * 1024,            // intermediate_buffer assume 40MB SRAM buffer per stack, 2MB write buffer
                 false,                             // intermediate_overlap
                 512.0 * 6 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF_Cons_7:one HBM4 stack for activation buffer, compare with H200
static SystemConfig HBF_Cons_7 = SystemConfig(
                 "HBF_Cons_7",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.45 * 1000 * 1000 * 1000 * 1000,  // memory_bandwidth
                 2.45 * 1000 * 1000 * 1000 * 1000,  // memory_read_bandwidth
                 73.40 * 1000 * 1000 * 1000,         // memory_write_bandwidth
                 1.6 * 1000 * 1000 * 1000 * 1000,  // intermediate_bandwidth, use one HBM4 stack
                 36.0 * 1 * 1024 * 1024 * 1024,     // intermediate_buffer HBM4 stack capacity
                 true,                             // intermediate_overlap
                 512.0 * 7 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF_Cons_8:sufficient SRAM for activation buffer, compare with H200
static SystemConfig HBF_Cons_8 = SystemConfig(
                 "HBF_Cons_8",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.80 * 1000 * 1000 * 1000 * 1000,  // memory_bandwidth
                 2.80 * 1000 * 1000 * 1000 * 1000,  // memory_read_bandwidth
                 83.89 * 1000 * 1000 * 1000,         // memory_write_bandwidth
                 2.80 * 1000 * 1000 * 1000 * 1000,  // intermediate_bandwidth
                 38.0 * 8 * 1024 * 1024,            // intermediate_buffer assume 40MB SRAM buffer per stack, 2MB write buffer
                 false,                             // intermediate_overlap
                 512.0 * 8 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF: 1.6 TB/s stack BW, tR=1us, 27 planes/die, compare with HBM4-6.4Gbps
// HBF_7:one HBM4 stack for activation buffer
static SystemConfig HBF_7 = SystemConfig(
                 "HBF_7",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 11.2 * 1000 * 1000 * 1000 * 1000,   // memory_bandwidth
                 11.2 * 1000 * 1000 * 1000 * 1000,   // memory_read_bandwidth
                 112.0 * 1000 * 1000 * 1000,           // memory_write_bandwidth
                 1.6 * 1000 * 1000 * 1000 * 1000,   // intermediate_bandwidth, use one HBM4-6.4Gbps stack
                 36.0 * 1 * 1024 * 1024 * 1024,     // intermediate_buffer HBM4 stack capacity
                 true,                              // intermediate_overlap
                 512.0 * 7 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF_8: sufficient SRAM for activation buffer
static SystemConfig HBF_8 = SystemConfig(
                 "HBF_8",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 12.8 * 1000 * 1000 * 1000 * 1000,   // memory_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000,   // memory_read_bandwidth
                 128.0 * 1000 * 1000 * 1000,         // memory_write_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000,   // intermediate_bandwidth, use SRAM buffer
                 38.0 * 8 * 1024 * 1024,            // intermediate_buffer assume 40MB SRAM buffer per stack, 2MB write buffer
                 false,                              // intermediate_overlap
                 512.0 * 8 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );
static SystemConfig HBF_7_high_write = SystemConfig(
                 "HBF_7_high_write",                // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 11.2 * 1000 * 1000 * 1000 * 1000,   // memory_bandwidth
                 11.2 * 1000 * 1000 * 1000 * 1000,   // memory_read_bandwidth
                 11.2 * 1000 * 1000 * 1000 * 1000,   // memory_write_bandwidth
                 1.6 * 1000 * 1000 * 1000 * 1000,   // intermediate_bandwidth, use one HBM4-6.4Gbps stack
                 36.0 * 1 * 1024 * 1024 * 1024,     // intermediate_buffer HBM4 stack capacity
                 true,                              // intermediate_overlap
                 512.0 * 7 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );

// HBF_8: sufficient SRAM for activation buffer
static SystemConfig HBF_8_high_write = SystemConfig(
                 "HBF_8_high_write",                      // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 100.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 1800.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 4000.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 12.8 * 1000 * 1000 * 1000 * 1000,   // memory_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000,   // memory_read_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000,   // memory_write_bandwidth
                 12.8 * 1000 * 1000 * 1000 * 1000,   // intermediate_bandwidth, use SRAM buffer
                 38.0 * 8 * 1024 * 1024,            // intermediate_buffer assume 40MB SRAM buffer per stack, 2MB write buffer
                 false,                              // intermediate_overlap
                 512.0 * 8 * 1024 * 1024 * 1024,      // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 6,                                 // num_cube
                 6                                  // int num_logic_cube
                 );
}  // namespace llm_system