from dataclasses import dataclass
import torch
import math
import os


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 8888  # Fixed vocab size as specified
    n_layer: int = 12      # Larger model (GPT2-medium level)
    n_head: int = 12       # More attention heads
    n_embed: int = 768     # Larger embedding dimension
    dropout: float = 0.1   # Slightly lower dropout for larger model
    bias: bool = False     # No bias terms (better performance)
    use_rotary: bool = True  # Use rotary embeddings
    batch_size: int = 12  # Add this line

    def __init__(self):
        self.base_batch_size = 12  # Base batch size per GPU

    @property
    def batch_size(self):
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        return self.base_batch_size * world_size


@dataclass
class TrainingConfig:
    def __init__(self):
        self.base_batch_size = 12  # Base batch size per GPU
        self.batch_size: int = self.base_batch_size * int(os.environ.get('WORLD_SIZE', 1))
        self.learning_rate: float = 5e-4    # Lower learning rate for stability
        self.max_iters: int = 100000       # More iterations for larger dataset
        self.weight_decay: float = 1e-1
        self.beta1: float = 0.9
        self.beta2: float = 0.95
        self.grad_clip: float = 1.0

        self.decay_lr: bool = True
        self.warmup_iters: int = 2000      # Longer warmup for stability
        self.lr_decay_iters: int = 100000  # Decay over full training
        self.min_lr: float = 5e-5         # Lower minimum learning rate

        self.eval_interval: int = 1000
        self.log_interval: int = 10
        self.eval_iters: int = 200
        self.gradient_accumulation_steps: int = 4  # More gradient accumulation steps

        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype: str = "bfloat16"  # Using bfloat16 for training stability
        self.compile: bool = True

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