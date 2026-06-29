#pragma once
#include <iostream>

#include "common/type.h"

namespace llm_system {

struct Stat {
  time_ns time;
  time_ns latency;
  time_ns queueing_delay = 0;
  time_ns arrival_time = 0;
  int seq_queue_size = 0;

  int process_token = 0;
  int batchsize = 0;
  int sum_seq = 0;
  int gen_seq = 0;
  int average_seq_len = 0;

  int input_len = 0;
  int output_len = 0;
  int num_sum_iter = 0;
  double sum_attention_opb = 0.0;
  int end_token = 0;

  time_ns qkv_gen = 0;
  time_ns atten_sum = 0;
  time_ns atten_gen = 0;
  time_ns o_proj = 0;
  time_ns lm_head = 0;
  time_ns ffn = 0;
  time_ns expert_ffn = 0;  // Includes internal sync wait (expert.cpp sync_*)
  time_ns communication = 0;

  // Previously missing breakdown items:
  // decode_kv_write: disagg decode KV bulk write stall (decode_kv_write.cpp)
  // pipeline_recv_wait: PP inter-stage wait time (communication.cpp PipelineRecv)
  // embedding_comm: embedding TP all-reduce (embedding.cpp)
  time_ns decode_kv_write = 0;
  time_ns pipeline_recv_wait = 0;

  // for MLA //
  time_ns q_down_proj = 0;
  time_ns kv_down_proj = 0;
  time_ns kr_proj = 0;
  time_ns q_up_proj = 0;
  time_ns qr_proj = 0;

  // MLA (Base) //
  time_ns kv_up_proj = 0;
  
  // MLA (Absorb) //
  time_ns tr_k_up_proj = 0;
  time_ns v_up_proj = 0;

  time_ns rope = 0;
  time_ns layernorm = 0;
  time_ns residual = 0;

  energy_nJ act_energy = 0;
  energy_nJ read_energy = 0;
  energy_nJ write_energy = 0;
  energy_nJ all_act_energy = 0;
  energy_nJ all_read_energy = 0;
  energy_nJ all_write_energy = 0;
  energy_nJ mac_energy = 0;
  energy_nJ total_energy = 0;

  energy_nJ FC_DRAM_energy = 0;
  energy_nJ FC_COMP_energy = 0;
  energy_nJ Attn_DRAM_energy = 0;
  energy_nJ Attn_COMP_energy = 0;
  energy_nJ MoE_DRAM_energy = 0;
  energy_nJ MoE_COMP_energy = 0;

  bool isOOM = false;

  // Disagg decode-only metrics
  time_ns second_token_time = 0;  // First decode iteration completion time
  time_ns token_interval = 0;    // Average inter-token time (excl. first decode)

  // Single trace steady state TPS metrics
  long long generated_tokens = 0;
  long long cumulative_generated_tokens = 0;
  double tps = 0.0;
  double tps_per_gpu = 0.0;
  int seeded_remaining = 0;
  int first_departure_seen = 0;   // 1 if first seeded departure detected
  int post_departure_steps = 0;   // decode steps after first departure
  double three_year_pec = 0.0;

  int is_mixed = 0; // wheter it is mixed stage or not
  int split = 0;
  std::string type;  // t2t, t2ft, e2e
  int iter_info;
};

}  // namespace llm_system
