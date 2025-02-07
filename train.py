from model import GPT
from config import GPTConfig, TrainingConfig
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import Task
import wandb

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def save_checkpoint(model, optimizer, iter_num, best_val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    
    # Save checkpoint
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        # Log best model to wandb
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_iter"] = iter_num
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iter_num:07d}.pt")
    
    torch.save(checkpoint, checkpoint_path)
    
    # Cleanup old checkpoints - keep only 3 most recent
    if not is_best:
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        )
        while len(checkpoints) > 3:
            os.remove(os.path.join(checkpoint_dir, checkpoints[0]))
            checkpoints.pop(0)

train_config = TrainingConfig()
out_dir = "out/"
checkpoint_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
resume = False

# Initialize wandb
wandb.init(
    project="backgpt",
    config={
        # Model config
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
    name=f"backgpt_l{GPTConfig.n_layer}_h{GPTConfig.n_head}_e{GPTConfig.n_embed}",
)

tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * GPTConfig.block_size
)
print("Tokens per iteration: ", tokens_per_iter)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

ctx = torch.autocast(train_config.device, dtype=torch.bfloat16)

model_args = dict(
    n_layer=GPTConfig.n_layer,
    n_head=GPTConfig.n_head,
    n_embed=GPTConfig.n_embed,
    block_size=GPTConfig.block_size,
    bias=GPTConfig.bias,
    vocab_size=GPTConfig.vocab_size,
    dropout=GPTConfig.dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=GPTConfig.block_size,
    device=train_config.device,
    num_workers=0,
)

best_val_loss = 1e9
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(train_config.device)
    print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(train_config.device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
)

if resume:
    optimizer.load_state_dict(checkpoint["optimizer"])

if train_config.compile:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()
while True:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
        )
        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)
        
        # Log GPU memory usage
        allocated, reserved = get_gpu_memory()
        writer.add_scalar("gpu_memory_allocated_gb", allocated, iter_num)
        writer.add_scalar("gpu_memory_reserved_gb", reserved, iter_num)
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Log to wandb
        wandb_metrics = {
            "train/loss": losses["train"],
            "val/loss": losses["val"],
            "train/lr": lr,
            "system/gpu_memory_allocated": allocated,
            "system/gpu_memory_reserved": reserved,
        }
        
        # Only add timing after first iteration
        if iter_num > 0:
            wandb_metrics["system/iteration_time_ms"] = dt * 1000
            
        wandb.log(wandb_metrics, step=iter_num)

        # Save checkpoint every 2 hours (7200 seconds)
        current_time = time.time()
        if not hasattr(save_checkpoint, 'last_save_time'):
            save_checkpoint.last_save_time = current_time
        
        if current_time - save_checkpoint.last_save_time > 7200:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, checkpoint_dir)
            save_checkpoint.last_save_time = current_time

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(model, optimizer, iter_num, best_val_loss, checkpoint_dir, is_best=True)
            print(f"Saved best checkpoint with val_loss {best_val_loss:.4f}")

    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % train_config.log_interval == 0:
        lossf = loss.item() * train_config.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        # Log more frequent training metrics to wandb
        wandb.log({
            "train/batch_loss": lossf,
            "train/tokens_per_second": tokens_per_iter / dt,
        }, step=iter_num)

    iter_num += 1

    if iter_num > train_config.max_iters:
        break

writer.close()
# Close wandb run
wandb.finish()
