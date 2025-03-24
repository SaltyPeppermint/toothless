from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelArguments:
    output_dir: str = field(default="model")
    d_model: int = field(default=128)
    num_layers: int = field(default=2)
    embedding_size: int = field(default=128)
    dim_feed_forward: int = field(default=128)
    dropout: float = field(default=0.2)
    # bf16: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: Path = field(metadata={"help": "Path to the training data."})
    split_size: float = field(default=0.9)
    random_state: int = field(default=42)
    batch_size: int = field(default=8)
    max_rel_pos: int = field(default=100)
    max_src_len: int = field(default=1000)
    max_tgt_len: int = field(default=1000)


@dataclass
class TrainingArguments:
    num_train_epochs: int = field(default=1)
    eval_each_epoch: bool = field(default=True)
    save_model_end: bool = field(default=True)
    tmax: int = field(default=2)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    logging_steps: int = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.01)
