# LLM Alignment Pipeline: SFT, DPO, GRPO, and PPO Components

A modular framework for adapting large language models through supervised fine-tuning, preference optimization, and reinforcement learning-based alignment methods.

The implementation focuses on reproducible LLM engineering workflows, including configurable training recipes, conversational dataset transformation, LoRA-based fine-tuning, reward-driven optimization, experiment tracking, and model artifact management.

## Highlights

- End-to-end LLM alignment pipeline implementing SFT, DPO, and GRPO workflows with PPO components
- Parameter-efficient fine-tuning using LoRA, QLoRA, and PEFT
- Dataset preprocessing utilities for instruction, preference, and reasoning tasks
- YAML-based configuration system for reproducible experiments
- Experiment tracking with Weights & Biases
- Hugging Face Hub-compatible model and adapter export
- Dockerized environment for reproducible execution

## System Architecture

The repository separates model loading, data processing, training logic, and artifact management into independent modules.

### 1. Model Initialization

- Hugging Face Transformers model loading
- Optional Unsloth acceleration for efficient training and inference

### 2. Dataset Processing

- Instruction formatting for SFT
- Preference pair conversion for DPO
- Reward-compatible prompt generation for GRPO
- Generation and reward computation utilities for PPO experiments

### 3. Training Pipeline

- Shared trainer construction patterns using Hugging Face TRL
- LoRA-based parameter-efficient adaptation
- Configurable optimization and generation strategies

### 4. Experiment Management

- YAML-based experiment configuration
- Reproducible random seed control
- Weights & Biases tracking
- Checkpoint management and Hugging Face Hub export

## Training Modes

### Supervised Fine-Tuning (SFT)

Fine-tunes instruction-following models by maximizing the likelihood of target assistant responses from supervised instruction datasets.

Workflow:

instruction dataset
→ conversational formatting
→ supervised training
→ LoRA adapter

### Direct Preference Optimization (DPO)

Optimizes a model using preference pairs containing a prompt, preferred completion, and rejected completion.

Workflow:

preference dataset
→ prompt/chosen/rejected formatting
→ preference objective
→ policy optimization

### Group Relative Policy Optimization (GRPO)

Optimizes language models using relative rewards computed from groups of generated outputs.

Workflow:

prompt
→ multiple sampled completions
→ reward evaluation
→ policy update

## Proximal Policy Optimization (PPO)

Provides PPO building blocks for reinforcement learning-based language model alignment.

Implemented components include:

- Policy and reference model initialization
- Response generation
- Reward model scoring
- Token-level log probability computation
- PPO clipped objective with KL regularization

## Technical Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face TRL (SFT, DPO, GRPO trainers, and RLHF utilities)
- PEFT and LoRA
- Unsloth
- Hugging Face Hub
- Weights & Biases
- Docker
- YAML configuration management

## Repository Structure

- [scripts](scripts): runnable training and inference entrypoints for SFT, DPO, GRPO, and evaluation.
- [src](src): core implementation modules for model loading, formatting, PEFT, trainers, and reward logic.
  - [src/models](src/models): model builders for standard Transformers and Unsloth paths.
  - [src/peft](src/peft): LoRA and adapter configuration helpers.
  - [src/rlhf](src/rlhf): preference optimization and reinforcement learning components including DPO, GRPO, and PPO utilities.
  - [src/sft](src/sft): supervised fine-tuning formatting and trainer utilities.
  - [src/utils](src/utils): shared configuration, hub, and seed utilities.
- [configs](configs): YAML-based training, LoRA, and RLHF configuration files.
- [templates](templates): prompt rendering templates used in reasoning and instruction formatting.
- [notebooks](notebooks): exploratory notebooks and development experiments.
- [docker](docker): Docker build files for reproducible environments.

## Requirements

This project requires Python 3.10+ and the dependencies listed in [requirements.txt](requirements.txt).

Install them with:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd llm-alignment-pipeline
pip install -r requirements.txt
```

### 2. Configure your training run

Default training behavior is controlled by the YAML files under [configs](configs). The most relevant settings are:

- [configs/rlhf/sft.yaml](configs/rlhf/sft.yaml)
- [configs/rlhf/dpo.yaml](configs/rlhf/dpo.yaml)
- [configs/rlhf/grpo.yaml](configs/rlhf/grpo.yaml)
- [configs/peft/lora.yaml](configs/peft/lora.yaml)

### 3. Launch one of the training entrypoints

#### SFT

```bash
python scripts/train_sft.py --model <model-name> --dataset <dataset-name-or-path>
```

#### DPO

```bash
python scripts/train_dpo.py --model <model-name> --dataset <dataset-name-or-path>
```

#### GRPO

```bash
python scripts/train_grpo.py --model <model-name> --dataset <dataset-name-or-path>
```

## Inference

The repository provides inference utilities for evaluating trained adapters:

- load LoRA adapters from local checkpoints
- run chat-template formatted generation
- evaluate reasoning outputs with task-specific reward functions

Example:

```bash
python scripts/infer_grpo_math.py --adapter <path>
```

## Configuration

Key configuration files include:

- [configs/rlhf/sft.yaml](configs/rlhf/sft.yaml): default SFT model, optimizer, and output settings
- [configs/rlhf/dpo.yaml](configs/rlhf/dpo.yaml): default DPO model, beta coefficient, and output settings
- [configs/rlhf/grpo.yaml](configs/rlhf/grpo.yaml): GRPO formatting, reward, training, and generation parameters
- [configs/peft/lora.yaml](configs/peft/lora.yaml): adapter rank, alpha, dropout, and target module configuration

## Output Management

Training runs create local model artifacts in the configured output folder and optionally publish them to the Hugging Face Hub. Experiment metadata is recorded by Weights & Biases, making it easier to compare training runs and reproduce results.

## Docker

A Dockerfile is included for reproducible environment setup and containerized training execution.

### Build the image

```bash
docker build -f docker/Dockerfile -t llm-alignment-pipeline .
```

### Run the container

```bash
docker run --rm -it llm-alignment-pipeline
```

## Notes

This repository provides an implementation framework for experimenting with modern LLM alignment techniques.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).