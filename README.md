# LLM Fine-tuning Platform

A training script for fine-tuning open-source LLMs using LoRA and QLoRA with parameter-efficient methods (PEFT), experiment tracking via MLflow, and YAML-based configuration.

## Features

- **LoRA / QLoRA fine-tuning** using Hugging Face PEFT and TRL's `SFTTrainer`
- **4-bit quantization** via bitsandbytes (QLoRA) for reduced memory usage
- **YAML config support** -- define all training parameters in a config file, override any value with CLI flags
- **MLflow experiment tracking** -- parameters and artifacts logged automatically
- **Configurable LoRA** -- rank, alpha, dropout, and target modules

## Project Structure

```
llm-finetuning-platform/
  train.py                  # Main training script
  configs/
    llama3_lora.yaml        # Example config for Llama 3 LoRA/QLoRA
  requirements.txt          # Python dependencies
  tests/
    test_train.py           # Unit tests
  pyproject.toml            # Project metadata and tool config
  Dockerfile                # GPU-enabled container image
  .github/workflows/ci.yml  # CI pipeline (lint + test)
```

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (16GB+ VRAM recommended for 7-8B models)
- PyTorch 2.1+

### Installation

```bash
git clone https://github.com/sanketny8/llm-finetuning-platform.git
cd llm-finetuning-platform

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

#### Option 1: Using a YAML config file

```bash
python train.py --config configs/llama3_lora.yaml
```

The config file sets model name, dataset path, LoRA parameters, and training
hyperparameters. See `configs/llama3_lora.yaml` for the full schema.

#### Option 2: Using CLI flags

```bash
python train.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset data/train.json \
    --output_dir ./outputs/llama3-lora \
    --use_qlora \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 2e-4
```

#### Option 3: Config file with CLI overrides

CLI flags take precedence over values in the config file:

```bash
python train.py \
    --config configs/llama3_lora.yaml \
    --learning_rate 1e-4 \
    --epochs 5
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--model_name` | Hugging Face model name or local path | Required |
| `--dataset` | Path to JSON training data | Required |
| `--output_dir` | Directory for saved model and checkpoints | `./outputs` |
| `--use_qlora` | Enable 4-bit QLoRA quantization | `False` |
| `--lora_r` | LoRA rank | `16` |
| `--lora_alpha` | LoRA alpha scaling | `32` |
| `--lora_dropout` | LoRA dropout rate | `0.05` |
| `--batch_size` | Per-device training batch size | `4` |
| `--epochs` | Number of training epochs | `3` |
| `--learning_rate` | Learning rate | `2e-4` |
| `--max_seq_length` | Maximum sequence length | `2048` |
| `--gradient_accumulation_steps` | Gradient accumulation steps | `4` |

### Monitoring

Training metrics are logged to MLflow. Launch the UI with:

```bash
mlflow ui
# Open http://localhost:5000
```

## Configuration File Format

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B
  load_in_4bit: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_dropout: 0.05

training:
  output_dir: ./outputs/llama3-lora
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_seq_length: 2048

optimization:
  optim: adamw_torch
  lr_scheduler_type: cosine

dataset:
  train_file: data/train.json
  text_field: text
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Docker

```bash
docker build -t llm-finetuning .
docker run --gpus all llm-finetuning \
    python3 train.py --config configs/llama3_lora.yaml
```

## Roadmap

The following features are planned but not yet implemented:

- Evaluation pipeline (standard benchmarks via lm-eval-harness)
- Multi-GPU training with DeepSpeed / FSDP
- Model merging (merge LoRA adapters back into base model)
- Weights & Biases integration
- Deployment scripts for vLLM / TGI inference servers
- Training recipes for common tasks (instruction tuning, chat, domain adaptation)

## License

MIT License -- see [LICENSE](LICENSE).

## Author

**Sanket Nyayadhish**
- Twitter: [@Ny8Sanket](https://twitter.com/Ny8Sanket)
- LinkedIn: [ny8sanket](https://linkedin.com/in/ny8sanket)
