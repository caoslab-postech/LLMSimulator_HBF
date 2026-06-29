#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <iostream>

#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

int main(int argc, char *argv[]) {
  YAML::Node config;
  if(argc > 1){
    std::string config_path = argv[1];
    config = YAML::LoadFile(config_path);
  }
  else{
    config = YAML::LoadFile("config.yaml");
  }

  std::string model_name = config["model"]["model_name"].as<std::string>();
  std::string processor_type =
      config["system"]["processor_type"].as<std::string>();
  std::string gpu_gen_name = config["system"]["gpu_gen"].as<std::string>();
  std::string processor_label = processor_type;
  if (processor_type.find("GPU") != std::string::npos) {
    processor_label += "_" + gpu_gen_name;
  }

  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();

  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();

  int max_batch_size = config["serving"]["max_batch_size"].as<int>();

  int max_process_token = config["serving"]["max_process_token"].as<int>();
  std::string output_path = config["log"]["output_directory"].as<std::string>();

  SystemConfig system_config;
  if(config["system"]["gpu_gen"].as<std::string>() == "A100"){
    system_config = A100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "H100"){
    system_config = H100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "H200"){
    system_config = H200;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B100"){
    system_config = B100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B200"){
    system_config = B200;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "R100"){
    system_config = R100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBM4"){
    system_config = HBM4;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_Cons_5"){
    system_config = HBF_Cons_5;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_Cons_6"){
    system_config = HBF_Cons_6;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_Cons_7"){
    system_config = HBF_Cons_7;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_Cons_8"){
    system_config = HBF_Cons_8;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_7"){
    system_config = HBF_7;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_8"){
    system_config = HBF_8;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_7_high_write"){
    system_config = HBF_7_high_write;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "HBF_8_high_write"){
    system_config = HBF_8_high_write;
  }
  else{
    fail("No GPU generation information");
  }

  // NVLink Config // 
  if(config["system"]["nvlink_gen"].as<int>() == 4){
    system_config.device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen (H100)
    system_config.device_ict_latency = 0.3 * 1000; // ns
  }
  else if(config["system"]["nvlink_gen"].as<int>() == 5){
    system_config.device_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen (B100, B200)
    system_config.device_ict_latency = 0.3 * 1000; // ns
  }
  else if(config["system"]["nvlink_gen"].as<int>() == 6){
    system_config.device_ict_bandwidth = 1800.0 * 1000 * 1000 * 1000; // B/s NVLink 6th Gen (B100, B200)
    system_config.device_ict_latency = 0.3 * 1000; // ns
  }
  else{
    fail("Not support NVLink generation");
  }

  // InfiniBand Config // 
  if(config["system"]["infiniband_gen"].as<int>() == 400){
    system_config.node_ict_bandwidth = 50.0 * 1000 * 1000 * 1000; // B/s Infiniband NDR
    system_config.node_ict_latency = 3000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 800){
    system_config.node_ict_bandwidth = 100.0 * 1000 * 1000 * 1000; // B/s InfiniBand XDR
    system_config.node_ict_latency = 3000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 3600){
    system_config.node_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 7200){
    system_config.node_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else{
    fail("Not support InfiniBand generation");
  }
  
  system_config.num_node = num_node;
  system_config.num_device = num_device;


  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;


  system_config.parallel_execution =
      config["system"]["optimization"]["parallel_execution"].as<bool>();
  system_config.hetero_subbatch =
      config["system"]["optimization"]["hetero_subbatch"].as<bool>();
  system_config.disagg_system =
      config["system"]["optimization"]["disagg_system"].as<bool>(); 
  system_config.use_low_unit_moe_only =
      config["system"]["optimization"]["use_low_unit_moe_only"].as<bool>();      
  system_config.use_ramulator =
      config["system"]["optimization"]["use_ramulator"].as<bool>();

  system_config.use_flash_mla =
      config["system"]["optimization"]["use_flash_mla"].as<bool>();
  system_config.use_flash_attention =
      config["system"]["optimization"]["use_flash_attention"].as<bool>();

  // kv cache reuse
  system_config.reuse_kv_cache =
      config["system"]["optimization"]["reuse_kv_cache"].as<bool>();
  system_config.kv_cache_reuse_rate =
      config["system"]["optimization"]["kv_cache_reuse_rate"].as<double>();
  
  // prefill mode or decode mode
  system_config.prefill_mode =
      config["system"]["optimization"]["prefill_mode"].as<bool>();
  system_config.decode_mode =
      config["system"]["optimization"]["decode_mode"].as<bool>();
  assertTrue((system_config.prefill_mode == false) || (system_config.decode_mode == false), 
            "prefill mode and decode mode is incompatible");

  assertTrue((system_config.parallel_execution == false) || (system_config.use_low_unit_moe_only == false),
            "parallel_execution & use_low_unit_moe_only are not compatible");
            
  if(system_config.prefill_mode){
    std::cout << "[Prefill Mode] Output Length is modified into 1" << std::endl;
  }

  // Single trace steady state mode
  system_config.single_trace_steady_state =
      config["simulation"]["single_trace_steady_state"]
      ? config["simulation"]["single_trace_steady_state"].as<bool>()
      : false;

  if(system_config.decode_mode){
    std::cout << "[Decode Mode] Current Length of sequences is modified into input_len" << std::endl;
  }

  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::GPU;
  } else if (!processor_type.compare("LOGIC")) {
    system_config.processor_type = {ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::LOGIC;
    system_config.low_processor_type = ProcessorType::LOGIC;
  } else if (!processor_type.compare("GPU+LOGIC")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::LOGIC;
    // system_config.parallel_execution = true;
  } else if (!processor_type.compare("GPU+PIM")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::PIM};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::PIM;
  }

  std::string expert_file_path;

  ModelConfig model_config;

  if (!model_name.compare("mixtral")) {
    model_config = mixtral;
  } else if (!model_name.compare("openMoE")) {
    model_config = openMoE;
  } else if (!model_name.compare("llama7bMoE")) {
    model_config = llama7bMoE;
  } else if (!model_name.compare("llama3_405B")) {
    model_config = llama3_405B;
  } else if (!model_name.compare("grok1")) {
    model_config = grok1;
  } else if (!model_name.compare("deepseekV3")) {
    model_config = deepseekV3;
  }else if (!model_name.compare("llama4_scout")) {
    model_config = llama4_scout;
  }else if (!model_name.compare("llama4_maverick")) {
    model_config = llama4_maverick;
  } 
  else {
    fail("No model configuration of " + model_name);
  }

  model_config.e_tp_dg =
      config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  model_config.ne_tp_dg =
      config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();

  // Pipeline parallelism degree (default 1 for backward compatibility)
  model_config.pp_dg = config["system"]["distribution"]["pipeline_degree"]
      ? config["system"]["distribution"]["pipeline_degree"].as<int>()
      : 1;
  
  model_config.compressed_kv =
      config["system"]["optimization"]["compressed_kv"].as<bool>();
  model_config.use_absorb =
      config["system"]["optimization"]["use_absorb"].as<bool>();
  model_config.skewness =
      config["simulation"]["skewness"].as<double>();
  
  model_config.precision_byte = config["simulation"]["precision_byte"].as<int>();
  if(model_config.precision_byte == 1){ // if FP8 or INT8 
    system_config.compute_peak_flops *= 2; // system_config has FP16 peak FLOPS information
  }

  system_config.exit_out_of_memory = config["simulation"]["exit_out_of_memory"].as<bool>();
  system_config.mem_cap_limit = config["simulation"]["mem_cap_limit"].as<bool>();
  system_config.ignore_activation_buffer_oom =
      config["simulation"]["ignore_activation_buffer_oom"]
          ? config["simulation"]["ignore_activation_buffer_oom"].as<bool>()
          : false;

  model_config.dataset = data_name;
  // Always seed global sequence lengths from YAML.
  // Static KV-cache sizing and memory checks use model_config lengths,
  // while trace mode still uses per-sequence Lin/Lout for execution.
  model_config.input_len = input_len;
  model_config.output_len = output_len;

  if (!data_name.compare("synthesis")) {
    expert_file_path = "none";
  } else {
    expert_file_path =
        "../expert_data/experts_" + model_name + "_" + data_name + ".csv";
  }

  if((system_config.decode_mode == true) && (output_len <= 1)){
    fail("[Decode Mode] Output length must be larger than 1");
  }

  // long max_batch_size = 128;
  if (max_process_token == 0) {
    // max_process_token = 8192 * 16;
    max_process_token = 65536 * 8;
  }
  int ne_tp_dg = model_config.ne_tp_dg;
  
  // DP degree accounts for both TP and PP dimensions.
  int ne_dp_dg = system_config.num_device * num_node
                 / (ne_tp_dg * model_config.pp_dg);
  int batch_size_per_dp = max_batch_size / ne_dp_dg;
  int num_max_batched_token = std::max(8192, batch_size_per_dp);
  if (num_max_batched_token != 8192) {
    std::cerr << "Adjust num_max_batched_token to " << num_max_batched_token
              << " to cover batch_size_per_dp=" << batch_size_per_dp
              << std::endl;
  }

  // Validate PP partition BEFORE model construction.
  // get_stage_layer_counts() internally asserts on invalid configs.
  if (model_config.pp_dg > 1) {
    auto stage_counts = get_stage_layer_counts(model_config);
    std::cerr << "PP partition (" << model_config.pp_dg << " stages): ";
    int layer_offset = 0;
    for (int i = 0; i < model_config.pp_dg; i++) {
      int count = stage_counts[i];
      std::cerr << "stage" << i << "=[" << layer_offset << ","
                << (layer_offset + count) << ")";
      if (i == 0) std::cerr << "+Emb";
      if (i == model_config.pp_dg - 1) std::cerr << "+LmHead";
      std::cerr << " ";
      layer_offset += count;
    }
    std::cerr << std::endl;
  }

  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, expert_file_path,
                        max_batch_size, num_max_batched_token,
                        max_process_token);

  // Single trace steady state validation
  if (system_config.single_trace_steady_state) {
    assertTrue(system_config.decode_mode,
        "single_trace_steady_state requires decode_mode");
    assertTrue(system_config.disagg_system,
        "single_trace_steady_state requires disagg_system");
    if (data_name != "synthesis") {
      assertTrue(scheduler->getUniqueTraceCount() == 1,
          "single_trace_steady_state requires exactly 1 unique trace sequence");
    }
    std::cerr << "Single-trace steady state: batch=" << scheduler->batch_size_per_dp
              << " output_len=" << model_config.output_len
              << " (uniformly seeded output progress + random expert routing)"
              << std::endl;
  }

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  bool out_of_memory = cluster->checkMemorySize();
  cluster->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = iter;

  int precision_bytes = model_config.precision_byte;

  std::string file_name;
  if(system_config.prefill_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_prefill.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_prefill.csv";
    }
  }
  else if(system_config.decode_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_decode.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_decode.csv";
    }
  }
  else{
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_label +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_PP" +
                            std::to_string(model_config.pp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + ".csv";
    }
  }
  if (out_of_memory && config["simulation"]["exit_out_of_memory"].as<bool>()) {
    std::cout << "Out of Memory: " << file_name << std::endl;
    return 0;
  }
  scheduler->getActualArrivalTime(total_iter);
  stat_list = cluster->runIteration(total_iter, file_name);
  // TopModuleGraph::Ptr top1 = cluster->get_device(8)->top_module_graph;
  std::string gantt_file_path =
      config["log"]["gantt_directory"].as<std::string>();

  if(config["log"]["export_gantt"].as<bool>()){
    cluster->exportGantt(gantt_file_path);
  }

  if(config["log"]["print_log"].as<bool>()){
    if (model_config.pp_dg > 1) {
      for (int pp = 0; pp < model_config.pp_dg; pp++) {
        int repr = pp * model_config.ne_tp_dg;
        std::cout << "=== Stage " << pp << " (device " << repr << ") ===" << std::endl;
        cluster->get_device(repr)->top_module_graph->print_timeboard();
      }
    } else {
      TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
      top0->print_timeboard();
    }
  }

  return 0;
}
