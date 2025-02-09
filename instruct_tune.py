import torch
import json
import requests
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from model import GPT
from config import GPTConfig, TrainingConfig
from tokenizer import Tokenizer
import wandb
from torch.nn import functional as F
import numpy as np
import os

# Constants
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DATA_DIR = Path("data")
INSTRUCT_FILE = DATA_DIR / "dolly.jsonl"
CHECKPOINT_DIR = Path("out/instruct_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def download_dataset():
    """Download Dolly dataset if not exists"""
    if not INSTRUCT_FILE.exists():
        print("Downloading Dolly dataset...")
        response = requests.get(DOLLY_URL, stream=True)
        with open(INSTRUCT_FILE, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)

class InstructDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Initializing InstructDataset with max_length={max_length}")
        
        print("Processing instruction dataset...")
        with open(INSTRUCT_FILE, 'r') as f:
            for line in tqdm(f):
                example = json.loads(line)
                context = example.get('context', '').strip()
                instruction = example['instruction'].strip()
                response = example['response'].strip()
                
                if context:
                    full_text = f"Context: {context}\n\nInstruction: {instruction}\n\nResponse: {response}"
                else:
                    full_text = f"Instruction: {instruction}\n\nResponse: {response}"
                
                # Reverse the entire sequence
                all_words = full_text.split()
                reversed_text = " ".join(all_words[::-1])
                
                tokens = self.tokenizer.encode(reversed_text, bos=True, eos=True)
                
                # Print debug info for long sequences
                if len(tokens) > self.max_length:
                    print(f"Skipping long sequence: {len(tokens)} tokens")
                    continue
                    
                # Ensure we have at least 2 tokens (for x and y)
                if len(tokens) < 2:
                    print(f"Skipping too short sequence: {len(tokens)} tokens")
                    continue
                
                self.examples.append(tokens)
                    
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Ensure we have enough tokens
        if len(tokens) < 2:
            print(f"Warning: sequence {idx} too short: {len(tokens)}")
            # Pad with at least 2 tokens
            tokens = tokens + [self.tokenizer.pad_id] * (2 - len(tokens))
            
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input/target pairs
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Verify shapes
        assert x.shape[0] == self.max_length - 1, f"Expected x shape {self.max_length-1}, got {x.shape[0]}"
        assert y.shape[0] == self.max_length - 1, f"Expected y shape {self.max_length-1}, got {y.shape[0]}"
        
        # Verify values are within vocab size
        vocab_size = 4096  # From model config
        assert torch.all(x < vocab_size), f"Input contains token ids >= vocab_size ({vocab_size})"
        assert torch.all(y < vocab_size), f"Target contains token ids >= vocab_size ({vocab_size})"
        
        return x, y

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = [t.to(device) for t in batch]
            logits, loss = model(x, y)
            total_loss += loss.sum().item()
            total_tokens += y.numel()
    
    model.train()
    return total_loss / total_tokens

def save_checkpoint(model, optimizer, model_config, iter_num, loss, is_best=False):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_config.__dict__,
        "iter_num": iter_num,
        "loss": loss
    }
    
    # Save periodic checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{iter_num:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint separately
    if is_best:
        best_path = CHECKPOINT_DIR / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        
    # Keep only last 3 checkpoints to save space
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    for checkpoint in checkpoints[:-3]:
        checkpoint.unlink()

def train_step(model, batch, optimizer, grad_clip):
    """Single training step for instruction tuning"""
    x, y = batch
    
    # Add shape checks
    B, T = x.shape
    assert T == model.config.block_size - 1, f"Expected sequence length {model.config.block_size-1}, got {T}"
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss = loss.mean()
    
    loss.backward()
    
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return loss.item()

def main():
    # Set CUDA launch blocking for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Initialize wandb with the requested project name
    wandb.init(
        project="backgpt_instruct_tiny",
        config={
            "architecture": "GPT",
            "dataset": "dolly-15k",
            "learning_rate": 1e-5,
            "epochs": 1000,
            "batch_size": 32,
            "optimizer": "AdamW"
        }
    )
    
    # Load the pre-trained model
    print("Loading pre-trained model...")
    checkpoint = torch.load("out/checkpoints_tiny/best_checkpoint_tiny.pt", map_location="cuda")
    
    # Use the same configuration as the base model
    model_args = checkpoint['model_args']
    print(f"Model config: {model_args}")
    
    model = GPT(GPTConfig(**model_args))
    
    # Fix the state dict keys by removing '_orig_mod.' prefix
    state_dict = checkpoint["model"]
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k[10:]] = v
        else:
            fixed_state_dict[k] = v
    
    # Load the fixed state dict
    model.load_state_dict(fixed_state_dict)
    model.to("cuda")
    
    # Load tokenizer - make sure it matches the vocab size from model config
    tokenizer = Tokenizer("data/tok4096.model")
    
    # Download and prepare dataset
    print("Loading Dolly dataset...")
    download_dataset()
    
    # Make sure block_size matches the model's configuration
    block_size = model_args['block_size']
    vocab_size = model_args['vocab_size']
    print(f"Using block size: {block_size}, vocab size: {vocab_size}")
    
    # Verify tokenizer vocab size matches model
    assert tokenizer.vocab_size == vocab_size, \
        f"Tokenizer vocab size ({tokenizer.vocab_size}) doesn't match model ({vocab_size})"
    
    full_dataset = InstructDataset(tokenizer, max_length=block_size)
    
    # Verify a few random samples before training
    print("\nVerifying random samples:")
    for i in range(3):
        idx = np.random.randint(len(full_dataset))
        x, y = full_dataset[idx]
        print(f"Sample {i} - x shape: {x.shape}, y shape: {y.shape}")
        print(f"x range: [{x.min()}, {x.max()}], y range: [{y.min()}, {y.max()}]")
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Training configuration - use similar settings as base training
    train_config = TrainingConfig()
    train_config.learning_rate = 1e-5  # Lower learning rate for fine-tuning
    train_config.max_iters = 1000
    train_config.batch_size = 32  # Smaller batch size for fine-tuning
    train_config.eval_interval = 100
    train_config.early_stop_patience = 5
    train_config.grad_clip = 1.0  # Same as base training
    
    # Dataloaders with smaller batch size for fine-tuning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Avoid potential CUDA issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Optimizer
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
        "cuda"
    )
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    pbar = tqdm(range(train_config.max_iters), desc="Training")
    train_iter = iter(train_loader)
    
    for iter_num in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y = batch
        # Add more shape/value checks
        assert x.shape[1] == block_size - 1, f"Wrong input sequence length: {x.shape[1]}"
        assert torch.all(x < vocab_size), "Input contains invalid token ids"
        assert torch.all(y < vocab_size), "Target contains invalid token ids"
        
        x = x.to("cuda")
        y = y.to("cuda")
        
        # Training step
        loss = train_step((x, y), model, optimizer, train_config.grad_clip)
        
        # Evaluation
        if (iter_num + 1) % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, "cuda")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, model_args, iter_num, val_loss, is_best=True)
            else:
                patience_counter += 1
            
            # Log metrics
            wandb.log({
                "train/loss": loss,
                "val/loss": val_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/iter": iter_num,
            })
            
            pbar.set_postfix(train_loss=f"{loss:.4f}", val_loss=f"{val_loss:.4f}")
            
            # Save periodic checkpoint
            save_checkpoint(model, optimizer, model_args, iter_num, loss)
            
            # Early stopping
            if patience_counter >= train_config.early_stop_patience:
                print(f"\nEarly stopping triggered after {iter_num + 1} iterations")
                break
        else:
            # Regular training logging
            wandb.log({
                "train/loss": loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/iter": iter_num,
            })
            pbar.set_postfix(loss=f"{loss:.4f}")
    
    # Save final model
    final_val_loss = evaluate(model, val_loader, "cuda")
    save_checkpoint(model, optimizer, model_args, train_config.max_iters, final_val_loss)
    
    wandb.finish()

if __name__ == "__main__":
    main() 