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
FINE_CHECKPOINT_DIR = OUT_DIR / "checkpoints_fine"
CHAT_CHECKPOINT_DIR = OUT_DIR / "checkpoints_chat"
CHAT_CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0


def save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=False):
    """Save model checkpoint"""
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
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    if is_best:
        checkpoint_path = CHAT_CHECKPOINT_DIR / "best_checkpoint_chat.pt"
        wandb.run.summary["best_val_loss_chat"] = best_val_loss
        wandb.run.summary["best_iter_chat"] = iter_num
    else:
        checkpoint_path = CHAT_CHECKPOINT_DIR / f"checkpoint_chat_{iter_num:07d}.pt"
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    # Cleanup old checkpoints - keep only 3 most recent
    if not is_best:
        checkpoints = sorted([f for f in os.listdir(CHAT_CHECKPOINT_DIR) if f.startswith("checkpoint_chat_")])
        while len(checkpoints) > 3:
            os.remove(CHAT_CHECKPOINT_DIR / checkpoints[0])
            checkpoints.pop(0)

def load_fine_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model_args = checkpoint['model_args']
    model = GPT(GPTConfig(**model_args)).to(device)
    model.load_state_dict(checkpoint['model'])
    optimizer = model.configure_optimizers(
        weight_decay=0.1,  # or load from checkpoint if you want
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        device_type=device.type,
    )
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint.get('iter_num', 0), checkpoint.get('best_val_loss', float('inf'))

def main():
    train_config = TrainingConfig()
    device = torch.device(train_config.device)
    ctx = torch.autocast(device.type, dtype=torch.bfloat16)

    # Load best checkpoint from fineweb training
    best_ckpt = FINE_CHECKPOINT_DIR / "best_checkpoint_fine.pt"
    model, optimizer, start_iter, best_val_loss = load_fine_checkpoint(best_ckpt, device)

    # Initialize wandb for chat finetuning
    wandb.init(
        project="backgpt",
        config={
            # Model config
            "dataset": "chat",
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embed": model.config.n_embed,
            "block_size": model.config.block_size,
            "vocab_size": model.config.vocab_size,
            "dropout": model.config.dropout,
            # Training config
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "warmup_iters": train_config.warmup_iters,
            "max_iters": train_config.max_iters,
            "grad_clip": train_config.grad_clip,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "finetune_on": "chat",
        },
        name=f"backgpt_chat_finetune_l{model.config.n_layer}_h{model.config.n_head}_e{model.config.n_embed}",
    )
    print("Wandb initialized")

    tokens_per_iter = (
        train_config.gradient_accumulation_steps
        * train_config.batch_size
        * model.config.block_size
    )
    print(f"Tokens per iteration: {tokens_per_iter:,}")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == "float16"))

    # Dataset iterator for chat data
    iter_batches = partial(
        Task.iter_batches,
        batch_size=train_config.batch_size,
        max_seq_len=model.config.block_size,
        device=device,
        bin_dir=Path("data"),  # expects chat .bin files in data/
    )
    train_batch_iter = iter_batches(split="train")
    t0 = time.time()

    print("Dataset iterator created")

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

    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)

    print("Starting chat finetuning...")
    iter_num = 0
    while True:
        lr = train_config.get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        micro_losses = []
        for micro_step in range(train_config.gradient_accumulation_steps):
            X, Y = next(train_batch_iter)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps
                micro_losses.append(loss.item() * train_config.gradient_accumulation_steps)
            scaler.scale(loss).backward()

        avg_loss = sum(micro_losses) / len(micro_losses)

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
            print(f"iter {iter_num}: loss {avg_loss:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
            wandb.log({
                "train/batch_loss": avg_loss,
                "train/lr": lr,
                "train/tokens_per_second": tokens_per_iter / dt,
                "system/iteration_time_ms": dt * 1000,
            }, step=iter_num)

        if iter_num % train_config.eval_interval == 0:
            losses = estimate_loss()
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
                save_checkpoint(model, optimizer, iter_num, best_val_loss, is_best=True)
                print(f"Saved best chat checkpoint with val_loss {best_val_loss:.4f}")

        iter_num += 1

        if not hasattr(save_checkpoint, 'last_save_time'):
            save_checkpoint.last_save_time = time.time()
        if time.time() - save_checkpoint.last_save_time > 7200:
            save_checkpoint(model, optimizer, iter_num, best_val_loss)
            save_checkpoint.last_save_time = time.time()

        if iter_num > train_config.max_iters:
            break

    save_checkpoint(model, optimizer, iter_num, best_val_loss)
    wandb.finish()

if __name__ == '__main__':
    main()