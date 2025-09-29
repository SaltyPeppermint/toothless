from dataclasses import dataclass, field

from dataclass_wizard import JSONWizard


@dataclass
class ModelArgs(JSONWizard):
    output_dir: str = field(default="model")
    d_model: int = field(default=768, metadata={"help": "Hidden state dimension size."})
    num_layers: int = field(default=12)
    head_dim: int = field(default=64)
    dim_feed_forward: int = field(default=3072, metadata={"help": "Feed forward dimension size."})
    dropout: float = field(default=0.1)
    n_heads: int = field(default=12)
    bf16: bool = field(default=True)


@dataclass
class DataArgs(JSONWizard):
    data_path: str = field(metadata={"help": "Path to the training data."})
    cache_dir: str = field(default="cache")
    split_size: float = field(default=0.9, metadata={"help": "Train/Test split ratio."})
    rng_seed: int = field(default=42)
    max_len: int = field(default=256)
    n_samples: int | None = field(default=None)
    tokenizer_samples: int = field(default=1_000_000)
    force_reload: bool = field(default=False)


@dataclass
class TrainArgs(JSONWizard):
    epochs: int = field(default=4)
    batch_size: int = field(default=16)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    min_lr: float = field(default=1e-6)
    eval_each_epoch: bool = field(default=True)
    save_model_end: bool = field(default=True)
    logging_steps: int = field(default=1)
    run_log_dir: str | None = field(default=None)
    trace: bool = field(default=False)


@dataclass
class FullArgs(JSONWizard):
    train: TrainArgs
    data: DataArgs
    model: ModelArgs
    verbose: bool = field(default=False)


@dataclass
class InferArgs(JSONWizard):
    folder: str
    batch_size: int
    model_suffix: str
    n_train_data: int | None = field(default=None)
    n_eval_data: int | None = field(default=None)
    verbose: bool = field(default=False)
