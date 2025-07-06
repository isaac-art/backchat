import os
import time
import wandb
import torch
from dataset import Task
from pathlib import Path
from functools import partial

from model import GPT
from config import GPTConfig, TrainingConfig

# Constants
OUT_DIR = Path("out")
CHECKPOINT_DIR = OUT_DIR / "checkpoints_fine"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=False):
    """Save model checkpoint"""
    # Convert config to a regular dictionary for serialization
    model_args = dict(
        n_layer=model.config.n_layer,
        n_head=model.config.n_head,
        n_embed=model.config.n_embed,
        block_size=model.config.block_size,
        bias=model.config.bias,
        vocab_size=model.config.vocab_size,
        dropout=model.config.dropout,
    )
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,  # Use the dictionary instead of config.__dict__
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    
    # Save checkpoint
    if is_best:
        checkpoint_path = CHECKPOINT_DIR / "best_checkpoint_fine.pt"
        # Log best model to wandb
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_iter"] = iter_num
    else:
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_fine_{iter_num:07d}.pt"
    
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    # Cleanup old checkpoints - keep only 3 most recent
    if not is_best:
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_fine_")])
        while len(checkpoints) > 3:
            os.remove(CHECKPOINT_DIR / checkpoints[0])
            checkpoints.pop(0)

def main():
    train_config = TrainingConfig()
    
    # Initialize wandb with fine-specific config
    wandb.init(
        # mode="offline",
        project="backgpt",
        config={
            # Model config
            "dataset": "fine",
            "n_layer": GPTConfig.n_layer,
            "n_head": GPTConfig.n_head,
            "n_embed": GPTConfig.n_embed,
            "block_size": GPTConfig.block_size,
            "vocab_size": GPTConfig.vocab_size,
            "dropout": GPTConfig.dropout,
            # Training config
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "warmup_iters": train_config.warmup_iters,
            "max_iters": train_config.max_iters,
            "grad_clip": train_config.grad_clip,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        },
        name=f"backgpt_fine_l{GPTConfig.n_layer}_h{GPTConfig.n_head}_e{GPTConfig.n_embed}",
    )
    
    print("Wandb initialized")
    
    tokens_per_iter = (
        train_config.gradient_accumulation_steps
        * train_config.batch_size
        * GPTConfig.block_size
    )
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    
    # Set up training environment
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    device = torch.device(train_config.device)
    ctx = torch.autocast(device.type, dtype=torch.bfloat16)
    
    # Initialize model
    model = GPT(GPTConfig).to(device)
    print("Model initialized")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == "float16"))
    
    # Optimizer
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
        device,
    )
    
    # Dataset iterator
    iter_batches = partial(
        Task.iter_batches,
        batch_size=train_config.batch_size,
        device=device,
        max_seq_len=GPTConfig.block_size,
        bin_dir=Path("data/fine"),
    )
    
    print("Dataset iterator created")
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    train_batch_iter = iter_batches(split="train")
    t0 = time.time()
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                X, Y = next(iter_batches(split=split))
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # Compile model if requested
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    print("Starting training loop")
    while True:
        print(f"\n[DEBUG] Iteration {iter_num} start")

        # Set learning rate
        lr = train_config.get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        micro_losses = []
        for micro_step in range(train_config.gradient_accumulation_steps):
            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Loading batch...")
            X, Y = next(train_batch_iter)
            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Batch loaded.")

            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Forward pass...")
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps
                micro_losses.append(loss.item() * train_config.gradient_accumulation_steps)
            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Forward pass done.")

            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Backward pass...")
            scaler.scale(loss).backward()
            print(f"[DEBUG] Iter {iter_num} Micro-step {micro_step}: Backward pass done.")
        
        avg_loss = sum(micro_losses) / len(micro_losses)
        
        if train_config.grad_clip != 0.0:
            print(f"[DEBUG] Iter {iter_num}: Clipping gradients...")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            print(f"[DEBUG] Iter {iter_num}: Gradients clipped.")
        
        print(f"[DEBUG] Iter {iter_num}: Optimizer step...")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"[DEBUG] Iter {iter_num}: Optimizer step done.")
        
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % train_config.log_interval == 0:
            print(f"iter {iter_num}: loss {avg_loss:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
            wandb.log({
                "train/batch_loss": avg_loss,
                "train/lr": lr,
                "train/tokens_per_second": tokens_per_iter / dt,
                "system/iteration_time_ms": dt * 1000,
            }, step=iter_num)
        
        if iter_num % train_config.eval_interval == 0:
            print(f"[DEBUG] Iter {iter_num}: Starting evaluation...")
            losses = estimate_loss()
            print(f"[DEBUG] Iter {iter_num}: Evaluation done.")
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            allocated, reserved = get_gpu_memory()
            wandb.log({
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "train/lr": lr,
                "system/gpu_memory_allocated": allocated,
                "system/gpu_memory_reserved": reserved,
            }, step=iter_num)
            
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                print(f"[DEBUG] Iter {iter_num}: Saving best checkpoint...")
                save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=True)
                print(f"[DEBUG] Iter {iter_num}: Best checkpoint saved.")

        iter_num += 1
        
        if not hasattr(save_checkpoint, 'last_save_time'):
            save_checkpoint.last_save_time = time.time()
        if time.time() - save_checkpoint.last_save_time > 7200:
            print(f"[DEBUG] Iter {iter_num}: Saving regular checkpoint...")
            save_checkpoint(model, optimizer, iter_num, best_val_loss)
            save_checkpoint.last_save_time = time.time()
            print(f"[DEBUG] Iter {iter_num}: Regular checkpoint saved.")
        
        if iter_num > train_config.max_iters:
            print(f"[DEBUG] Iter {iter_num}: Max iterations reached, breaking loop.")
            break

        print(f"[DEBUG] Iteration {iter_num} end\n")
    
    # Final save
    save_checkpoint(model, optimizer, iter_num, best_val_loss)
    wandb.finish()

if __name__ == '__main__':
    main() 