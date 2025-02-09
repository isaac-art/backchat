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
                if len(tokens) <= self.max_length:
                    self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
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

def main():
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
    checkpoint = torch.load("out/best_checkpoint.pt", map_location="cuda")
    model_config = GPTConfig(**checkpoint["model_args"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to("cuda")
    
    # Load tokenizer
    tokenizer = Tokenizer("data/tok4096.model")
    
    # Download and prepare dataset
    download_dataset()
    full_dataset = InstructDataset(tokenizer, max_length=model_config.block_size)
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Training configuration
    train_config = TrainingConfig()
    train_config.learning_rate = 1e-5
    train_config.max_iters = 1000
    train_config.batch_size = 32
    train_config.eval_interval = 100
    train_config.early_stop_patience = 5
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True
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
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = tuple(t.to("cuda") for t in batch)
        
        # Training step
        loss = train_step(model, batch, optimizer, train_config.grad_clip)
        
        # Evaluation
        if (iter_num + 1) % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, "cuda")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, model_config, iter_num, val_loss, is_best=True)
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
            save_checkpoint(model, optimizer, model_config, iter_num, loss)
            
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
    save_checkpoint(model, optimizer, model_config, train_config.max_iters, final_val_loss)
    
    wandb.finish()

if __name__ == "__main__":
    main() 