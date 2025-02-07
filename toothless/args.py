from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-7B")
    output_dir: str = field(default="output_qwen")
    bf16: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str | None = field(default="cache/start.parquet", metadata={"help": "Path to the training data."})
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)
    lazy_preprocess: bool = field(default=False)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    eval_each_epoch: bool = field(default=True)
    save_model_end: bool = field(default=True)
    tmax: int = field(default=30)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    logging_steps: int = field(default=1)
    # ds_config: str = field(metadata={"help": "Deepspeed config is required"})
    # report_to: Optional[str] = field(default="none")
    # gradient_checkpointing: bool = False
    # save_strategy: Optional[str] = field(default="steps")
    # save_steps: Optional[int] = field(default=1000)
    # save_total_limit: Optional[int] = field(default=10)
    # warmup_ratio: Optional[float] = field(default=0.01)
    # lr_scheduler_type: Optional[str] = field(default="cosine")
