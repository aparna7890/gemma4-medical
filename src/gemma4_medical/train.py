"""
Training module for Gemma 4 E2B medical reasoning fine-tuning.
Supports LoRA and QLoRA configurations via Unsloth.
"""
import os
import gc
import torch
import wandb
from datasets import load_dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from typing import Optional


def load_and_format_dataset(tokenizer, train_size: int = 5000, eval_size: int = 100):
    """Load and format the medical SFT dataset."""
    ds = load_dataset(
        "FreedomIntelligence/medical-o1-reasoning-SFT", 
        "en", 
        split="train"
    )

    train_ds = ds.select(range(train_size))
    eval_ds  = ds.select(range(len(ds) - eval_size, len(ds)))

    def format_example(example):
        full_answer = (
            f"{example['Complex_CoT']}\n\n"
            f"**Final Answer:** {example['Response']}"
        )
        messages = [
            {"role": "user",      
             "content": [{"type": "text", "text": example["Question"]}]},
            {"role": "assistant", 
             "content": [{"type": "text", "text": full_answer}]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    train_formatted = train_ds.map(format_example, remove_columns=ds.column_names)
    eval_formatted  = eval_ds.map(format_example,  remove_columns=ds.column_names)
    return train_formatted, eval_formatted


def build_model(
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    load_in_4bit: bool = True,
    max_seq_length: int = 512,
):
    """Load base model and attach LoRA adapters."""
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def run_training(
    run_name: str,
    lora_r: int = 16,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    lora_dropout: float = 0.0,
    max_steps: int = 200,
    output_base: str = "/content/gemma4-checkpoints",
    wandb_project: str = "gemma4-medical",
):
    """
    Full training pipeline for one sweep run.
    Loads model fresh each call to avoid GPU state leakage between runs.
    """
    set_seed(3407)
    torch.cuda.empty_cache()

    # Build model
    model, tokenizer = build_model(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Load dataset
    train_formatted, eval_formatted = load_and_format_dataset(tokenizer)

    # W&B
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "lora_dropout": lora_dropout,
            "max_steps": max_steps,
        },
        reinit=True,
    )

    # Trainer
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir                  = output_dir,
        max_steps                   = max_steps,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        per_device_eval_batch_size  = 1,
        learning_rate               = learning_rate,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        eval_strategy               = "steps",
        eval_steps                  = 50,
        save_strategy               = "steps",
        save_steps                  = 100,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        logging_steps               = 10,
        report_to                   = "wandb",
        seed                        = 3407,
        max_seq_length              = 512,
        dataset_text_field          = "text",
        packing                     = False,
        fp16                        = True,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_formatted,
        eval_dataset  = eval_formatted,
        args          = training_args,
        callbacks     = [
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.005,
            )
        ],
    )

    stats = trainer.train()

    # Save adapter
    adapter_path = f"/content/lora-adapter-{run_name}"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Cleanup
    wandb.finish()
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return stats