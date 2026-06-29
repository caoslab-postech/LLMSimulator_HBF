#pragma once
#include <chrono>  // for timing of loading sequence
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/type.h"

namespace llm_system {

struct SequenceInfo {
  using Ptr = std::shared_ptr<SequenceInfo>;
  int Lin;
  int Lout;
  int cur;
  std::vector<std::vector<int>> expert_list;
};

class Sequence {
 public:
  using Ptr = std::shared_ptr<Sequence>;

  inline static int count_id = 0;

  [[nodiscard]] static Ptr Create(int expert_seq_id = 0, int input_len = 256,
                                  int output_len = 256) {
    return Ptr(new Sequence(expert_seq_id, input_len, output_len));
  }

  // Sequence operator=(Sequence seq) {
  //   Sequence temp;
  //   temp.input_len = seq.input_len;
  //   temp.max_len = seq.max_len;
  //   temp.num_process_token = seq.num_process_token;
  //   temp.current_len = seq.current_len;
  //   return temp;
  // }

  int id;
  int expert_seq_id;
  int input_len;
  int output_len;
  int total_len;
  int num_process_token;
  int current_len;

  int num_sum_iter;
  time_ns gen_start_time;
  time_ns arrival_time;
  time_ns first_token_time;
  time_ns end_token_time;
  time_ns queueing_delay;

  void update(time_ns time);
  void setGenStartTime(time_ns time);

  bool record;
  bool sum_stage;
  bool get_expert_from_list;

  // Disagg decode-only fields
  time_ns second_token_time;           // First decode iteration completion time
  int prompt_reused_len;               // KV tokens already present (reuse rate)
  bool track_second_token_time;        // Only true for disagg+decode_mode sequences
  std::set<int> prompt_kv_written_devices;  // device_total_rank set: KV write done per device

  // dowon revised
  // PP decode microbatch fields (decode-only mode).
  // Enable per-sequence token-level scheduling without a global round barrier.
  time_ns next_decode_ready_time = 0; // Absolute time the sequence is ready to launch its next token
  int microbatch_id = -1;             // Persistent microbatch assignment (-1 = unassigned)
  bool in_flight = false;             // Currently being processed in a microbatch execution
  time_ns last_decode_finish_time = 0; // Absolute finish time of the most recent token

  // Single trace steady state fields.
  // Initial seeded batch is used for TPS observation only, not latency.
  bool seeded_steady_state = false;   // Member of initial seeded batch
  bool suppress_latency = false;      // Skip t2ft/e2e row emission
  int initial_generated_output = 0;   // Pre-seeded output token count at init
  bool turnover_seed = false;         // Counts toward seeded_remaining for TPS end

 private:
  Sequence(int expert_seq_id, int input_len, int output_len);
  std::vector<std::vector<int>> expert_list;  // 2차원 벡터 type,sequence안의
                                              // token 수만큼 expert들이 존재.
};

class Scheduler;
class BatchedSequence {
 public:
  // static int cur;
  using Ptr = std::shared_ptr<BatchedSequence>;
  using Scheduler_ptr = std::shared_ptr<Scheduler>;

  [[nodiscard]] static Ptr Create(int num_expert = 8, int top_k = 2,
                                  Scheduler_ptr scheduler = nullptr) {
    return Ptr(new BatchedSequence(num_expert, top_k, scheduler));
  }

  BatchedSequence(int num_expert, int top_k, Scheduler_ptr scheduler)
      : num_expert(num_expert), top_k(top_k), scheduler(scheduler) {
    for (int i = 0; i < num_expert; i++) {  // initialize
      local_num_token_in_expert.push_back(0);
      num_token_in_expert.push_back(0);
    }
    cur_layer = 0;
  }

  void add(Sequence::Ptr seq);

  // Allocating snapshot API — returns a copy of the filtered/full sequence list.
  // Use only in erase-safe paths where pop()/erase() may be called during iteration.
  // Remaining callers after snapshot-churn optimization:
  //   scheduler.cpp updateScheduler()
  //   scheduler.cpp updateSchedulerSumGenSplit()
  std::vector<Sequence::Ptr> get_seq();
  std::vector<Sequence::Ptr> get_sum();
  std::vector<Sequence::Ptr> get_gen();

  // --- Non-allocating traversal API ---
  // These avoid snapshot vector allocation for read-only paths.
  // Sequence field mutation (e.g. seq->num_process_token = ...) is allowed,
  // but pop()/erase() on this BatchedSequence::sequence during iteration is NOT.
  // When using count_*() together with for_each_*() in the same function,
  // the callback must NOT change gen/sum membership fields (current_len, input_len)
  // — otherwise count and iteration results will be inconsistent.
  //
  // Usage rules:
  //   get_seq_ref() must be received as: const auto& seqs = ...get_seq_ref();
  //   for_each_* callback argument is const Sequence::Ptr& (shared_ptr by const ref).
  //   count_gen()/count_sum() are count-only; they never allocate a filtered vector.

  const std::vector<Sequence::Ptr>& get_seq_ref() const { return sequence; }

  template <class F>
  void for_each_seq(F&& f) const {
    for (const auto& seq : sequence) f(seq);
  }

  template <class F>
  void for_each_sum(F&& f) const {
    for (const auto& seq : sequence) {
      if (seq->current_len < seq->input_len) f(seq);
    }
  }

  template <class F>
  void for_each_gen(F&& f) const {
    for (const auto& seq : sequence) {
      if (seq->current_len >= seq->input_len) f(seq);
    }
  }

  bool has_sum() const {
    for (const auto& seq : sequence)
      if (seq->current_len < seq->input_len) return true;
    return false;
  }

  bool has_gen() const {
    for (const auto& seq : sequence)
      if (seq->current_len >= seq->input_len) return true;
    return false;
  }

  int count_sum() const {
    int n = 0;
    for (const auto& seq : sequence)
      if (seq->current_len < seq->input_len) ++n;
    return n;
  }

  int count_gen() const {
    int n = 0;
    for (const auto& seq : sequence)
      if (seq->current_len >= seq->input_len) ++n;
    return n;
  }

  void update_expert(int num_expert, int top_k, bool need_new_expert);
  void clear_expert();
  void update(time_ns time);
  void pop(Sequence::Ptr seq);
  int get_num_seq();
  int get_process_token();
  int64_t get_gen_process_token();
  int64_t get_sum_process_token();
  int get_average_sequence_length();
  int64_t get_total_sequence_length();

  void add_dummy_sequence(int num_seq, int input_len, int output_len);

  void add_sequence(std::vector<int> seq_ids);

  int cur_layer;  // 이 batch의 현재 layer와 stage
  int num_expert;
  int top_k;
  std::vector<int> local_num_token_in_expert;
  std::vector<int> num_token_in_expert;
  std::vector<Sequence::Ptr> sequence;
  std::vector<int> seq_ids;
  Scheduler_ptr scheduler;
};

}  // namespace llm_system
