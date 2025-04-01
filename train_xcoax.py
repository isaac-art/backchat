import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
from torch.cuda.amp import autocast, GradScaler
import logging

from model import GPT
from config_xcoax import GPTConfig, TrainingConfig

# Ensure deterministic behavior
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_distributed():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def get_batch(data, config, device_type="cuda"):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64)) for i in ix])
    
    if device_type == "cuda":
        x, y = x.pin_memory().to(device=config.device, non_blocking=True), y.pin_memory().to(device=config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

def get_data():
    data_dir = Path("data/xcoax")
    train_data = []
    val_data = []
    
    # Load all binary files
    bin_files = sorted(list(data_dir.glob("chunk_*.bin")))
    split_idx = int(len(bin_files) * 0.9)  # 90% train, 10% val
    
    print("Loading training data...")
    for file in bin_files[:split_idx]:
        with open(file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint16)
            train_data.append(data)
    train_data = np.concatenate(train_data)
    
    print("Loading validation data...")
    for file in bin_files[split_idx:]:
        with open(file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint16)
            val_data.append(data)
    val_data = np.concatenate(val_data)
    
    return train_data, val_data

def evaluate(model, val_data, config, max_batches=None):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(config.eval_iters):
            X, Y = get_batch(val_data, config)
            logits, loss = model(X, Y)
            losses.append(loss.item())
            
    model.train()
    return np.mean(losses)

def save_checkpoint(model, optimizer, train_config, iter_num, best_val_loss, checkpoint_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_config': train_config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    print(f"Saving checkpoint to {checkpoint_dir}/ckpt_{iter_num}.pt")
    torch.save(checkpoint, f"{checkpoint_dir}/ckpt_{iter_num}.pt")

def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU Memory Available: {gpu_memory / 1e9:.2f} GB")
        return gpu_memory
    return None

def reduce_gradients(model):
    """Reduce gradients across all GPUs."""
    if not isinstance(model, DDP):
        return
    
    for param in model.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= torch.distributed.get_world_size()

def get_data_for_worker():
    """Get data shard for current GPU worker"""
    train_data, val_data = get_data()
    
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        # Shard the data
        train_len = len(train_data) // world_size
        val_len = len(val_data) // world_size
        
        train_start = rank * train_len
        val_start = rank * val_len
        
        train_data = train_data[train_start:train_start + train_len]
        val_data = val_data[val_start:val_start + val_len]
    
    return train_data, val_data

def setup_logging(local_rank):
    """Setup logging for distributed training"""
    logging.basicConfig(
        format=f'[{local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO if local_rank == 0 else logging.WARNING
    )
    
    return logging.getLogger(__name__)

def train():
    # Initialize distributed training if needed
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        local_rank = setup_distributed()
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master_process = local_rank == 0
    else:
        local_rank = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True
    
    # Initialize configs
    model_config = GPTConfig()
    train_config = TrainingConfig()
    train_config.device = device
    
    # Initialize model
    if master_process:
        print("Initializing model...")
    model = GPT(model_config)
    model.to(train_config.device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # Load data
    if master_process:
        print("Loading data...")
    train_data, val_data = get_data_for_worker()
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    if master_process:
        wandb.init(project="xcoax-gpt", name="100bt-pretrain")
    
    scaler = GradScaler()
    
    logger = setup_logging(local_rank)
    logger.info("Starting training...")
    
    while True:
        # Get batch and compute loss
        t0 = time.time()
        X, Y = get_batch(train_data, model_config, device_type=train_config.device)
        
        # Forward pass
        with autocast():
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        # Update weights
        if (iter_num + 1) % train_config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        
        if iter_num % train_config.log_interval == 0 and master_process:
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/lr": train_config.get_lr(iter_num),
            })
        
        # Evaluate on validation set
        if iter_num > 0 and iter_num % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_data, model_config)
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint_dir = Path("out/xcoax")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    if master_process:
                        save_checkpoint(model, optimizer, train_config, iter_num, best_val_loss, checkpoint_dir)
            
            if master_process:
                wandb.log({
                    "iter": iter_num,
                    "val/loss": val_loss,
                })
        
        # Update learning rate
        if train_config.decay_lr:
            lr = train_config.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        iter_num += 1
        
        # End training if we've reached max_iters
        if iter_num > train_config.max_iters:
            break
    
    if ddp:
        destroy_process_group()

def main():
    ddp = int(os.environ.get('RANK', -1)) != -1
    try:
        train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        if ddp:
            destroy_process_group()

if __name__ == '__main__':
    main() 