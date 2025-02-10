from dataclasses import dataclass
import torch
import math

@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 12000
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.1
    bias: bool = False
    use_rotary: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 48
    learning_rate: float = 3e-4
    max_iters: int = 100000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 4000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5

    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 5

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