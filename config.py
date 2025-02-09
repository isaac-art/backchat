from dataclasses import dataclass
import torch
import math


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = None  # Will be set based on dataset
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 512
    dropout: float = 0.15
    bias: bool = False
    use_rotary: bool = True

    def __post_init__(self):
        # Set vocab size based on dataset if not explicitly provided
        if self.vocab_size is None:
            self.vocab_size = 50257  # Default for OWT2


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 7e-4
    max_iters: int = 50000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 7e-5

    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 2

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True

    def get_lr(self, it: int) -> float:
        """Get learning rate at iteration it according to schedule."""
        # 1) Linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
