#pragma once
#include <map>
#include <string>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "model/llm.h"
#include "model/model_config.h"
#include "module/module.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class Model : public Module {
 public:
  Model(ModelConfig& model_config, Cluster::Ptr cluster,
        Scheduler::Ptr scheduler)
      : Module(),
        model_config(model_config),
        cluster(cluster),
        scheduler(scheduler) {
    for (int i = 0; i < cluster->num_total_device; i++) {
      device = cluster->get_device(i);
      device->setModelConfig(model_config);
      Module::Ptr top_module = std::static_pointer_cast<Module>(
          LLM::Create(model_config, cluster, scheduler, device));

      add_module(top_module);

      model_distribute(top_module, cluster, scheduler, i);
      device->reset_timeboard();
      device->reset_status();
    }
  };

 private:
  void model_distribute(Module::Ptr top_module, Cluster::Ptr cluster,
                        Scheduler::Ptr scheduler, int device_rank) {
    BatchedSequence::Ptr max_metadata = scheduler->getMaxMetadata(model_config.num_routed_expert, model_config.top_k);

    // Stage 0 receives token indices; other stages receive hidden activations.
    // Graph tracing needs the correct input shape per stage.
    RankInfo ri = decompose_rank(device_rank,
        model_config.ne_tp_dg, model_config.pp_dg);

    std::vector<int> input_shape;
    std::string input_name;
    if (ri.pp_rank == 0) {
        input_shape = {1, scheduler->model_config.n_vocab};
        input_name = "EmbeddingVector";
    } else {
        input_shape = {1, scheduler->model_config.hidden_dim};
        input_name = "PipelineActivation";
    }

    Tensor::Ptr input_tensor = Tensor::Create(
        input_name, input_shape, "act",
        cluster->get_device(device_rank),
        scheduler->model_config.precision_byte);
    (*top_module)(input_tensor, max_metadata);
  };

  Cluster::Ptr cluster;
  Scheduler::Ptr scheduler;
  ModelConfig model_config;
};

}  // namespace llm_system