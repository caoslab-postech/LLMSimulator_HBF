#include "model/llm.h"

#include "module/communication.h"
#include "module/decode_kv_write.h"
#include "module/decoder.h"
#include "module/embedding.h"
#include "module/lm_head.h"
#include "module/layer.h"

namespace llm_system {

LLM::LLM(const ModelConfig& model_config, Cluster::Ptr cluster,
         Scheduler::Ptr scheduler, Device::Ptr device)
    : Module("", "LLM", device), model_config(model_config),
      pipeline_recv(nullptr), pipeline_send(nullptr),
      decode_kv_write(nullptr) {

  int pp_dg = model_config.pp_dg;
  int ne_tp_dg = model_config.ne_tp_dg;
  RankInfo ri = decompose_rank(device->device_total_rank, ne_tp_dg, pp_dg);
  stage_id = ri.pp_rank;
  auto [ls, le] = get_stage_layer_range(model_config, stage_id);
  layer_start = ls;
  layer_end = le;

  // TP group: ne_tp_dg contiguous devices. Used for non-MoE modules
  // (embedding, attention, dense FFN, lm_head, AllReduce).
  std::vector<int> tp_device_list;
  set_device_list(tp_device_list, ri.tp_group_offset, ne_tp_dg);

  // MoE stage group: all devices in the same PP stage across all DP replicas.
  // Non-contiguous when dp > 1. Used for MoEScatter, MoEGather, Route, expert hosting.
  // Example: pp=0, ne_tp=8, pp_dg=2, dp=2 → {0..7, 16..23}
  int dp_degree = cluster->num_total_device / (ne_tp_dg * pp_dg);
  std::vector<int> moe_stage_device_list =
      get_stage_device_list(stage_id, ne_tp_dg, pp_dg, dp_degree);

  // Stage 0 only: Embedding layer
  if (stage_id == 0) {
    auto embedding_layer = Embedding::Create(module_map_name, "Embedding_layer",
                                             model_config, tp_device_list, device);
    add_module(embedding_layer);
  }

  // Non-first stages: receive activation from previous stage via P2P
  if (stage_id > 0) {
    int src_device = get_pp_peer(device->device_total_rank, ne_tp_dg, -1);
    // PipelineSend's module_map_name is known at construction time because
    // LLM always registers it as "pipeline_send" under the "::LLM" prefix.
    std::string send_mname = "::LLM::pipeline_send";
    pipeline_recv = PipelineRecv::Create(
        module_map_name, "pipeline_recv", src_device,
        send_mname, "pipeline_activation",
        tp_device_list, device);
    add_module(pipeline_recv);
  }

  // Disagg decode-only: stage-local KV bulk write (one-time per query per device).
  // Placed before decoder layers so that first decode iteration includes KV write stall.
  if (device->config.disagg_system && device->config.decode_mode) {
    decode_kv_write = DecodeKVWrite::Create(
        module_map_name, "decode_kv_write", model_config,
        stage_id, layer_start, layer_end,
        tp_device_list, device);
    add_module(decode_kv_write);
  }

  // Only create this stage's decoder layers (not the full model)
  if (model_config.model_name != "deepseekV3") {
    for (int layer = layer_start; layer < layer_end; layer++) {
      if ((model_config.expert_freq != 0) &&
          (layer % model_config.expert_freq == 0)) {
        // MoEDecoder uses stage-wide device list for cross-DP expert routing.
        auto moe_decoder = MoEDecoder::Create(
            module_map_name, "MoE_decoder_" + std::to_string(layer),
            model_config, scheduler, moe_stage_device_list, device, layer);
        add_module(moe_decoder);
      } else {
        auto decoder = Decoder::Create(
            module_map_name, "decoder_" + std::to_string(layer),
            model_config, scheduler, tp_device_list, device, layer);
        add_module(decoder);
      }
    }
  } else if (model_config.model_name == "deepseekV3") {
    for (int layer = layer_start; layer < layer_end; layer++) {
      if (layer < model_config.first_k_dense) {
        auto decoder = Decoder::Create(
            module_map_name, "decoder_" + std::to_string(layer),
            model_config, scheduler, tp_device_list, device, layer);
        add_module(decoder);
      } else {
        auto moe_decoder = MoEDecoder::Create(
            module_map_name, "MoE_decoder_" + std::to_string(layer),
            model_config, scheduler, tp_device_list, device, layer);
        add_module(moe_decoder);
      }
    }
  } else {
    fail("No configuration of " + model_config.model_name);
  }

  // Non-last stages: send activation to next stage via P2P
  if (stage_id < pp_dg - 1) {
    int dst_device = get_pp_peer(device->device_total_rank, ne_tp_dg, +1);
    pipeline_send = PipelineSend::Create(
        module_map_name, "pipeline_send", dst_device,
        tp_device_list, device);
    add_module(pipeline_send);
  }

  // Last stage only: LmHead layer
  if (stage_id == pp_dg - 1) {
    auto lm_head = LmHead::Create(module_map_name, "lm_head",
                                   model_config, tp_device_list, device);
    add_module(lm_head);
  }
}

Tensor::Ptr LLM::forward(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr temp = input;
  Tensor::Ptr out;

  // Stage 0: token embedding
  if (stage_id == 0) {
    Module::Ptr embedding = get_module("Embedding_layer");
    temp = (*embedding)(temp, sequences_metadata);
  }

  // Non-first stages: wait for previous stage's activation
  if (stage_id > 0) {
    temp = (*pipeline_recv)(temp, sequences_metadata);
  }

  // Disagg decode-only: stage-local KV bulk write before decoder layers
  if (decode_kv_write) {
    temp = (*decode_kv_write)(temp, sequences_metadata);
  }

  // Execute only this stage's decoder layers
  for (int layer = layer_start; layer < layer_end; layer++) {
    Module::Ptr decoder;
    if (model_config.model_name == "deepseekV3") {
      if (layer < model_config.first_k_dense) {
        decoder = get_module("decoder_" + std::to_string(layer));
      } else {
        decoder = get_module("MoE_decoder_" + std::to_string(layer));
      }
    } else {
      if ((model_config.expert_freq != 0) &&
          (layer % model_config.expert_freq == 0)) {
        decoder = get_module("MoE_decoder_" + std::to_string(layer));
      } else {
        decoder = get_module("decoder_" + std::to_string(layer));
      }
    }
    // Chain hidden states through decoder layers (fix: propagate temp)
    temp = (*decoder)(temp, sequences_metadata);
  }
  out = temp;

  // Non-last stages: send activation to next stage
  if (stage_id < model_config.pp_dg - 1) {
    out = (*pipeline_send)(out, sequences_metadata);
  }

  // Last stage: project to vocabulary logits
  if (stage_id == model_config.pp_dg - 1) {
    Module::Ptr lm_head = get_module("lm_head");
    out = (*lm_head)(out, sequences_metadata);
  }

  return out;
}
};  // namespace llm_system
