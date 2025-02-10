from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    block_size: int = 1024  # Increased for longer context
    vocab_size: int = 8000  # Larger vocab for real-world text
    n_layer: int = 12       # Increased depth
    n_head: int = 12
    n_embed: int = 768      # Increased embedding size
    dropout: float = 0.1    # Slightly lower dropout for pretraining
    bias: bool = False
    use_rotary: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 48            # Adjusted for L40
    learning_rate: float = 3e-4     # Slightly lower for stability
    max_iters: int = 100000        # Longer training for larger dataset
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 4000       # Longer warmup
    lr_decay_iters: int = 100000   # Match max_iters
    min_lr: float = 3e-5

    eval_interval: int = 500       # Less frequent eval for faster training
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 5  # Increased for effective batch size

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True 