from pydantic_settings import BaseSettings
from typing import Literal

class TrainingConfig(BaseSettings):
    # Model
    model_name: str = "unsloth/gemma-4-E2B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: Literal["cosine", "linear"] = "cosine"

    # Eval & saving
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    max_new_tokens: int = 512

    # Paths
    output_dir: str = "/kaggle/working/gemma4-medical-checkpoints"
    adapter_save_path: str = "/kaggle/working/lora-adapter"

    # Reproducibility
    seed: int = 3407

    # W&B
    wandb_project: str = "gemma4-medical"
    run_name: str = "lora-r16-lr2e4-baseline"

    class Config:
        env_file = ".env"