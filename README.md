# LLMSimulator_HBF

## Overview

LLMSimulator_HBF is a C++-based simulator extended from LLMSimulator to evaluate HBF-based LLM-serving systems. This simulator implements HBF modeling on top of the original LLMSimulator framework and focuses on the decode node in prefill/decode disaggregated LLM-serving systems.

The current implementation supports Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) models. It models decode-stage execution behavior and evaluates how HBF-based memory configurations affect LLM-serving performance under different model, workload, and hardware settings.

The current HBF model is implemented at the LLMSimulator_HBF modeling level. Detailed HBF device modeling inside Ramulator2 is not implemented.

## Prerequisites

* g++ 11.4.0
* CMake
* clang++

## Getting Started

### Building LLMSimulator_HBF

1. Clone the repository.

```bash
git clone https://github.com/caoslab-postech/LLMSimulator_HBF.git
cd LLMSimulator_HBF
```

2. Build the simulator.

```bash
mkdir -p build
cd build
cmake ..
make -j
```

### How to run

LLMSimulator_HBF uses `config.yaml` as the main configuration file. Modify `config.yaml` according to the target model, workload, and hardware configuration, and then run the simulator from the build directory.

```bash
./run > test.log
```

## Note

This simulator was developed as part of our work, “Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges.”
