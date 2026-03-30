"""Main training script for LLM fine-tuning with LoRA/QLoRA."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import mlflow


def load_config(config_path: str) -> dict:
    """Load training configuration from a YAML file.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Dictionary with configuration values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments.

    If --config is provided, values from the YAML file are used as defaults.
    Explicit CLI flags always override config file values.
    """
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA/QLoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model name or path")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--use_qlora", action="store_true", default=None, help="Use QLoRA (4-bit)")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Resolve values: CLI flag > config file > hardcoded default
    resolved = argparse.Namespace(
        model_name=args.model_name or config.get("model", {}).get("name"),
        dataset=args.dataset or config.get("dataset", {}).get("train_file"),
        output_dir=args.output_dir or config.get("training", {}).get("output_dir", "./outputs"),
        use_qlora=(
            args.use_qlora
            if args.use_qlora is not None
            else config.get("model", {}).get("load_in_4bit", False)
        ),
        lora_r=args.lora_r or config.get("lora", {}).get("r", 16),
        lora_alpha=args.lora_alpha or config.get("lora", {}).get("lora_alpha", 32),
        lora_dropout=args.lora_dropout if args.lora_dropout is not None else config.get("lora", {}).get("lora_dropout", 0.05),
        batch_size=args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 4),
        epochs=args.epochs or config.get("training", {}).get("num_train_epochs", 3),
        learning_rate=args.learning_rate or config.get("training", {}).get("learning_rate", 2e-4),
        max_seq_length=args.max_seq_length or config.get("training", {}).get("max_seq_length", 2048),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            or config.get("training", {}).get("gradient_accumulation_steps", 4)
        ),
        # Additional fields from config only (no CLI flag)
        target_modules=config.get("lora", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        optim=config.get("optimization", {}).get("optim", "adamw_torch"),
        lr_scheduler_type=config.get("optimization", {}).get("lr_scheduler_type", "cosine"),
        text_field=config.get("dataset", {}).get("text_field", "text"),
    )

    # Validate required fields
    if not resolved.model_name:
        raise ValueError("--model_name is required (via CLI or config file)")
    if not resolved.dataset:
        raise ValueError("--dataset is required (via CLI or config file)")

    return resolved


def train(args):
    """Run the training loop.

    Args:
        args: Namespace with all training parameters.
    """
    # Start MLflow run
    mlflow.set_experiment("llm-finetuning")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # Configure quantization
        bnb_config = None
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for training
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        # Load dataset
        dataset = load_dataset("json", data_files=args.dataset, split="train")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            save_strategy="epoch",
            logging_steps=10,
            warmup_steps=100,
            optim=args.optim,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to=["mlflow"],
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            dataset_text_field=args.text_field,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model(args.output_dir)
        print(f"Model saved to {args.output_dir}")

        # Log model to MLflow
        mlflow.log_artifact(args.output_dir)


def main():
    """Entry point: parse arguments and start training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
