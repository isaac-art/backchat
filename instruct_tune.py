import torch
import json
import requests
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from model import GPT
from config import GPTConfig, TrainingConfig
from tokenizer import Tokenizer
import wandb
from torch.nn import functional as F

# Constants
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DATA_DIR = Path("data")
INSTRUCT_FILE = DATA_DIR / "dolly.jsonl"

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
        
        # Load and process the dataset
        with open(INSTRUCT_FILE, 'r') as f:
            for line in f:
                example = json.loads(line)
                # Get components
                context = example.get('context', '').strip()
                instruction = example['instruction'].strip()
                response = example['response'].strip()
                
                # Build the full text first
                if context:
                    full_text = f"Context: {context}\n\nInstruction: {instruction}\n\nResponse: {response}"
                else:
                    full_text = f"Instruction: {instruction}\n\nResponse: {response}"
                
                # Reverse the entire sequence
                all_words = full_text.split()
                reversed_text = " ".join(all_words[::-1])
                
                # Tokenize the reversed sequence
                tokens = self.tokenizer.encode(reversed_text, bos=True, eos=True)
                if len(tokens) <= self.max_length:
                    self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def train_step(model, batch, optimizer, grad_clip):
    x, y = batch
    logits, loss = model(x, y)
    loss = loss.mean()
    loss.backward()
    
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()

def main():
    # Initialize wandb
    wandb.init(project="backgpt-instruct", name="instruct_tune")
    
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
    dataset = InstructDataset(tokenizer, max_length=model_config.block_size)
    
    # Training configuration
    train_config = TrainingConfig()
    train_config.learning_rate = 1e-5  # Lower learning rate for fine-tuning
    train_config.max_iters = 1000      # Fewer iterations for fine-tuning
    train_config.batch_size = 32       # Smaller batch size
    
    # Optimizer
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
        "cuda"
    )
    
    # Training loop
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    model.train()
    pbar = tqdm(range(train_config.max_iters), desc="Training")
    train_iter = iter(train_loader)
    
    for iter_num in pbar:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move batch to device
        batch = tuple(t.to("cuda") for t in batch)
        
        # Train step
        loss = train_step(model, batch, optimizer, train_config.grad_clip)
        
        # Log metrics
        wandb.log({
            "train/loss": loss,
            "train/lr": optimizer.param_groups[0]["lr"],
        })
        pbar.set_postfix(loss=f"{loss:.4f}")
        
        # Save checkpoint periodically
        if (iter_num + 1) % 100 == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_config.__dict__,
                "iter_num": iter_num,
            }
            torch.save(checkpoint, "out/instruct_checkpoint.pt")
    
    # Save final model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_config.__dict__,
        "iter_num": train_config.max_iters,
    }
    torch.save(checkpoint, "out/instruct_final.pt")
    wandb.finish()

if __name__ == "__main__":
    main() 