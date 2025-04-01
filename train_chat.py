import os
import time
import wandb
import torch
from pathlib import Path
from functools import partial

from model import GPT
from config import GPTConfig, TrainingConfig
from dataset import Task

# Constants
DATA_DIR = Path("data")
OUT_DIR = Path("out")
CHECKPOINT_DIR = OUT_DIR / "checkpoints_chat"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model.config.__dict__,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    
    # Save checkpoint
    if is_best:
        checkpoint_path = CHECKPOINT_DIR / "best_checkpoint_chat.pt"
        # Log best model to wandb
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_iter"] = iter_num
    else:
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_chat_{iter_num:07d}.pt"
    
    torch.save(checkpoint, checkpoint_path)
    
    # Cleanup old checkpoints - keep only 3 most recent
    if not is_best:
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_chat_")])
        while len(checkpoints) > 3:
            os.remove(CHECKPOINT_DIR / checkpoints[0])
            checkpoints.pop(0)

def main():
    # Model configuration
    model_config = GPTConfig(
        block_size=1024,
        vocab_size=8192,  # Reduced from 32000
        n_layer=8,        # Reduced from 12
        n_head=8,         # Reduced from 12
        n_embed=512,      # Reduced from 768
        dropout=0.1,
        bias=False,
        use_rotary=True
    )
    
    # Training configuration
    train_config = TrainingConfig(
        batch_size=16,    # Reduced from 32
        learning_rate=6e-4,
        max_iters=50000,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        decay_lr=True,
        warmup_iters=2000,
        lr_decay_iters=50000,
        min_lr=6e-5,
        eval_interval=100,
        eval_iters=200,
        log_interval=10
    )
    
    # Initialize wandb
    wandb.init(
        project="backgpt_chat",
        config={
            # Model config
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embed": model_config.n_embed,
            "block_size": model_config.block_size,
            "vocab_size": model_config.vocab_size,
            "dropout": model_config.dropout,
            # Training config
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "warmup_iters": train_config.warmup_iters,
            "max_iters": train_config.max_iters,
            "grad_clip": train_config.grad_clip,
        }
    )
    
    
    # Initialize model
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT(model_config)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
        device
    )
    
    # Dataset iterator
    iter_batches = partial(
        Task.iter_batches,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.block_size,
        device=device,
        num_workers=0,
        dataset="chat"  # New dataset type
    )
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    train_batch_iter = iter_batches(split="train")
    t0 = time.time()
    
    while True:
        # Get the next batch
        try:
            batch = next(train_batch_iter)
        except StopIteration:
            train_batch_iter = iter_batches(split="train")
            batch = next(train_batch_iter)
        
        # Determine and set the learning rate for this iteration
        if train_config.decay_lr:
            lr = train_config.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Forward backward update
        x, y = batch
        logits, loss = model(x, y)
        loss = loss.mean()  # Collapse all losses if they are scattered
        
        # Backward pass
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % train_config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=iter_num)
        
        if iter_num > 0 and iter_num % train_config.eval_interval == 0:
            # Evaluate the model
            model.eval()
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                with torch.no_grad():
                    batch = next(iter_batches(split="val"))
                    X, Y = batch
                    logits, loss = model(X, Y)
                    losses[k] = loss.mean()
            val_loss = losses.mean()
            model.train()
            
            # Log validation metrics
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            wandb.log({
                "val/loss": val_loss,
            }, step=iter_num)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=True)
            
            # Regular checkpoint save
            if iter_num % 1000 == 0:
                save_checkpoint(model, optimizer, iter_num, val_loss)
        
        iter_num += 1
        
        # Termination conditions
        if iter_num > train_config.max_iters:
            break
    
    # Final save
    save_checkpoint(model, optimizer, iter_num, val_loss)
    wandb.finish()

if __name__ == '__main__':
    main() 