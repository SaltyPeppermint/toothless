from dataclasses import dataclass, field
from typing import Optional

from dataclass_wizard import JSONWizard


@dataclass
class ModelArguments(JSONWizard):
    disentangled: bool = field(metadata={"help": "Use the disentangled or vanilla model."})
    output_dir: str = field(default="model")
    d_model: int = field(default=768, metadata={"help": "Hidden state dimension size."})
    num_layers: int = field(default=12)
    dim_feed_forward: int = field(default=3072, metadata={"help": "Feed forward dimension size."})
    dropout: float = field(default=0.1)
    n_heads: int = field(default=12)
    anc_heads: int = field(default=8)
    sib_heads: int = field(default=4)
    pos_type: str = field(default="p2q_p2k", metadata={"help": "Position type seperated by underscore."})
    with_pos: bool = field(default=False, metadata={"help": "Enable rotary positional encoding for decoder."})


@dataclass
class DataArguments(JSONWizard):
    data_path: str = field(metadata={"help": "Path to the training data."})
    cache_dir: str = field(default="cache")
    split_size: float = field(default=0.9, metadata={"help": "Train/Test split ratio."})
    rng_seed: int = field(default=42)
    sample_distance: int = field(
        default=8, metadata={"help": "Minimum distance when splitting the explanation chains recursively."}
    )
    batch_size: int = field(default=16)
    k: int = field(default=15, metadata={"help": "Max relative positional distance."})
    max_len: int = field(default=192)
    force_reload: bool = field(default=False)
    sample_limit: int | None = field(default=None)
    sample_cache_dir: str | None = field(default=None)


@dataclass
class TrainingArguments(JSONWizard):
    epochs: int = field(default=4)
    eval_each_epoch: bool = field(default=True)
    save_model_end: bool = field(default=True)
    tmax: int = field(default=2)
    learning_rate: float = field(default=1e-5)
    min_lr: float = field(default=1e-6)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    logging_steps: int = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.01)
    bf16: bool = field(default=True)
    trace: bool = field(default=False)
    run_log_dir: str | None = field(default=None)


@dataclass
class TrainRunArgs:
    train: TrainingArguments
    data: DataArguments
    model: ModelArguments


@dataclass
class InferenceArguments(JSONWizard):
    infer_data: str
    folder: str
    n_train_data: int | None
    n_eval_data: int | None
    batch_size: int
    model_suffix: str
    bf16: bool = field(default=True)
    verbose: bool = field(default=False)
