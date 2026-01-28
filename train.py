"""Main training script for LLM fine-tuning."""

import argparse
import os
from pathlib import Path

import torch
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA/QLoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
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
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            fp16=True,
            save_strategy="epoch",
            logging_steps=10,
            warmup_steps=100,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to=["mlflow"],
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=2048,
            dataset_text_field="text",
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(args.output_dir)
        print(f"Model saved to {args.output_dir}")
        
        # Log model to MLflow
        mlflow.log_artifact(args.output_dir)


if __name__ == "__main__":
    main()

