from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    output_dir: Optional[str] = field(default="output_qwen")
    bf16: bool = True


@dataclass
class DataArguments:
    data_path: str | None = field(default="cache/start.parquet", metadata={"help": "Path to the training data."})
    test_size: Optional[float] = field(default=0.2)
    random_state: Optional[int] = field(default=42)
    update_cache: bool = False
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments:
    ds_config: str = field(metadata={"help": "Deepspeed config is required"})
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    evaluation_strategy: Optional[str] = field(default="no")
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=1000)
    save_total_limit: Optional[int] = field(default=10)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.1)
    adam_beta2: Optional[float] = field(default=0.95)
    warmup_ratio: Optional[float] = field(default=0.01)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    logging_steps: Optional[int] = field(default=1)
    report_to: Optional[str] = field(default="none")
    gradient_checkpointing: bool = False
    tmax: float = 0.2
