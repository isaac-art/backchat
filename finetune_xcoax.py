import os
import time
import torch
from pathlib import Path
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb

from model import GPT
from config_xcoax import GPTConfig, TrainingConfig

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

def get_instruction_data():
    data_dir = Path("data/instruct")
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

def evaluate(model, val_data, config):
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
    print(f"Saving checkpoint to {checkpoint_dir}/instruct_ckpt_{iter_num}.pt")
    torch.save(checkpoint, f"{checkpoint_dir}/instruct_ckpt_{iter_num}.pt")

def validate_checkpoint(checkpoint):
    required_keys = ['model', 'optimizer', 'train_config', 'iter_num']
    if not all(key in checkpoint for key in required_keys):
        raise ValueError("Checkpoint missing required keys")

def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Validate checkpoint structure
    required_keys = ['model', 'optimizer', 'train_config', 'iter_num']
    if not all(key in checkpoint for key in required_keys):
        raise ValueError("Checkpoint missing required keys")
    
    return checkpoint

def finetune(pretrained_model_path: str):
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
    
    # Initialize configs with adjusted parameters for finetuning
    model_config = GPTConfig()
    train_config = TrainingConfig()
    # Adjust training parameters for finetuning
    train_config.learning_rate = 1e-4  # Lower learning rate for finetuning
    train_config.min_lr = 1e-5
    train_config.warmup_iters = 1000
    train_config.max_iters = 20000
    train_config.lr_decay_iters = 20000
    train_config.eval_interval = 200
    train_config.device = device
    
    # Load pre-trained model
    if master_process:
        print(f"Loading pre-trained model from {pretrained_model_path}")
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    validate_checkpoint(checkpoint)
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
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
    
    # Load instruction tuning data
    if master_process:
        print("Loading instruction tuning data...")
    train_data, val_data = get_instruction_data()
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    if master_process:
        wandb.init(project="xcoax-gpt", name="instruction-tuning")
    
    while True:
        # Get batch and compute loss
        t0 = time.time()
        X, Y = get_batch(train_data, model_config, device_type=train_config.device)
        
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        # Update weights
        optimizer.step()
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Finetune XCOAX model on instruction dataset")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to pre-trained XCOAX model checkpoint")
    
    args = parser.parse_args()
    finetune(args.model_path) 