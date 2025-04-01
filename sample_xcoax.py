import time
import torch
import numpy as np
from pathlib import Path
from model import GPT
from config_xcoax import GPTConfig
from tokenizer import Tokenizer

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with same config
    model_config = GPTConfig()
    model = GPT(model_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    return model

def sample(
    model,
    tokenizer,
    prompt="",
    max_new_tokens=500,
    temperature=0.8,
    top_k=200,
    device='cuda',
    seed=1234
):
    """Sample from the model"""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Encode the prompt
    if prompt:
        prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
        x = (torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...])
    else:
        # If no prompt, start with just BOS token
        x = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Generate tokens
    y = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(x)
            logits = logits[:, -1, :] # Take last timestep
            
            # Apply temperature
            logits = logits / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            y.append(idx_next.item())
            
            # If we hit EOS token, stop
            if idx_next.item() == tokenizer.eos_id:
                break
            
            # Update context window
            x = torch.cat((x, idx_next), dim=1)
            
            # If context is too long, crop it
            if x.size(1) > model.config.block_size:
                x = x[:, -model.config.block_size:]
    
    return y

class ModelRunner:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        if self.device == 'cuda':
            self.check_gpu_memory()
        model = load_model(model_path)
        model.to(self.device)
        return model
    
    def check_gpu_memory(self, required_mb=4000):  # Adjust based on model size
        if torch.cuda.is_available():
            free_memory = (torch.cuda.get_device_properties(0).total_memory - 
                         torch.cuda.memory_allocated()) / (1024 * 1024)
            if free_memory < required_mb:
                raise RuntimeError(f"Not enough GPU memory. Need {required_mb}MB, have {free_memory:.0f}MB")
    
    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Load tokenizer
    tokenizer_path = Path("data") / "tok8888_xcoax.model"
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer(str(tokenizer_path))
    
    # Load latest model checkpoint
    checkpoint_dir = Path("out/xcoax")
    checkpoints = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    latest_checkpoint = checkpoints[-1]
    
    # Load model
    model_runner = ModelRunner(latest_checkpoint)
    
    # Interactive sampling loop
    print("\nEnter prompts for sampling (Ctrl+C to exit)")
    print("You can adjust sampling parameters with commands:")
    print("  temp=X - Set temperature (default 0.8)")
    print("  top_k=X - Set top-k (default 200)")
    print("  tokens=X - Set max tokens (default 500)")
    
    # Default parameters
    temperature = 0.8
    top_k = 200
    max_tokens = 500
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            # Check for parameter adjustment
            if prompt.startswith(("temp=", "top_k=", "tokens=")):
                try:
                    param, value = prompt.split("=")
                    value = float(value) if param == "temp" else int(value)
                    if param == "temp":
                        temperature = value
                    elif param == "top_k":
                        top_k = value
                    else:
                        max_tokens = value
                    print(f"Set {param} to {value}")
                except ValueError:
                    print("Invalid parameter value")
                continue
            
            # Generate
            t0 = time.time()
            y = sample(
                model_runner.model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=model_runner.device
            )
            t1 = time.time()
            
            # Decode and print
            generated = tokenizer.decode(y)
            print("\nGenerated text:")
            print("=" * 40)
            print(generated)
            print("=" * 40)
            print(f"\nGenerated {len(y)} tokens in {(t1-t0):.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main() 