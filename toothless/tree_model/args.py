from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelArguments:
    output_dir: str = field(default="model")
    d_model: int = field(default=256, metadata={"help": "Hidden state dimension size."})
    num_layers: int = field(default=8)
    embedding_size: int = field(default=256)
    dim_feed_forward: int = field(default=256, metadata={"help": "Feed forward dimension size."})
    dropout: float = field(default=0.2)
    anc_heads: int = field(default=3)
    sib_heads: int = field(default=1)
    pos_type: str = field(default="p2q_p2k", metadata={"help": "Position type seperated by underscore."})
    with_pos: bool = field(default=False, metadata={"help": "Enable rotary positional encoding for decoder."})
    bf16: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: Path = field(metadata={"help": "Path to the training data."})
    split_size: float = field(default=0.9, metadata={"help": "Train/Test split ratio."})
    random_state: int = field(default=42)
    # pairs_per_expl: int = field(default=4)
    min_expl_distance: int = field(
        default=8, metadata={"help": "Minimum distance when splitting the explanation chains recursively."}
    )
    batch_size: int = field(default=8)
    k: int = field(default=15, metadata={"help": "Max relative positional distance."})
    max_len: int = field(default=256)
    force_reload: bool = field(default=False)


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
