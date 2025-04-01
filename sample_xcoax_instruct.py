import time
import torch
import numpy as np
from pathlib import Path
from model import GPT
from config_xcoax import GPTConfig
from tokenizer import Tokenizer

def load_model(checkpoint_path):
    """Load the instruction-tuned model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with same config
    model_config = GPTConfig()
    model = GPT(model_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    return model

def format_response(response: str) -> str:
    """Format response for backwards generation"""
    # Clean response
    response = response.strip()
    
    # Split response into words and reverse
    response_words = response.split()
    reversed_response = " ".join(response_words[::-1])
    
    # Format with special tokens
    # For backwards generation, we provide the response and model generates instruction
    formatted = f"<|im_start|><|response|>{reversed_response}<|im_end|>"
    return formatted

def extract_instruction(generated_text: str) -> str:
    """Extract and un-reverse the instruction from generated text"""
    try:
        # Find instruction section
        start_idx = generated_text.find("<|instruction|>") + len("<|instruction|>")
        end_idx = generated_text.find("<|im_end|>", start_idx)
        if start_idx == -1 or end_idx == -1:
            return "Error: Could not find instruction markers"
        
        # Extract instruction
        instruction = generated_text[start_idx:end_idx].strip()
        
        # Un-reverse the instruction
        words = instruction.split()
        return " ".join(words[::-1])
    except Exception as e:
        return f"Error processing instruction: {str(e)}"

def sample(
    model,
    tokenizer,
    response: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda',
    seed: int = 1234
) -> str:
    """Sample from the instruction-tuned model - generating instruction for given response"""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Format and encode the response
    prompt = format_response(response)
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    x = (torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...])
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Generate tokens
    generated = []
    instruction_started = False
    instruction_ended = False
    
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
            token = idx_next.item()
            generated.append(token)
            
            # Check for instruction markers
            if not instruction_started:
                # Convert recent tokens to text to check for instruction start
                recent = tokenizer.decode(generated[-10:])  # Look at last few tokens
                if "<|instruction|>" in recent:
                    instruction_started = True
            
            if instruction_started and not instruction_ended:
                # Check if we've completed the instruction
                recent = tokenizer.decode(generated[-10:])
                if "<|im_end|>" in recent:
                    instruction_ended = True
                    break
            
            # Update context window
            x = torch.cat((x, idx_next), dim=1)
            
            # If context is too long, crop it
            if x.size(1) > model.config.block_size:
                x = x[:, -model.config.block_size:]
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated)
    
    # Extract and process instruction
    return extract_instruction(generated_text)

def main():
    # Load tokenizer
    tokenizer_path = Path("data") / "tok8888_xcoax.model"
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer(str(tokenizer_path))
    
    # Load latest instruction-tuned checkpoint
    checkpoint_dir = Path("out/xcoax")
    checkpoints = sorted(list(checkpoint_dir.glob("instruct_ckpt_*.pt")))
    if not checkpoints:
        raise ValueError(f"No instruction-tuned checkpoints found in {checkpoint_dir}")
    latest_checkpoint = checkpoints[-1]
    
    # Load model
    model = load_model(latest_checkpoint)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Interactive sampling loop
    print("\nEnter responses and get instructions (Ctrl+C to exit)")
    print("You can adjust sampling parameters with commands:")
    print("  temp=X - Set temperature (default 0.8)")
    print("  top_k=X - Set top-k (default 200)")
    print("  tokens=X - Set max tokens (default 200)")
    print("\nProvide a response and the model will generate an instruction that could have led to it")
    
    # Default parameters
    temperature = 0.8
    top_k = 200
    max_tokens = 200
    
    while True:
        try:
            response = input("\nResponse: ").strip()
            
            # Check for parameter adjustment
            if response.startswith(("temp=", "top_k=", "tokens=")):
                try:
                    param, value = response.split("=")
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
            
            # Generate instruction for the response
            t0 = time.time()
            instruction = sample(
                model,
                tokenizer,
                response,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            t1 = time.time()
            
            # Print instruction
            print("\nGenerated Instruction:")
            print("=" * 40)
            print(instruction)
            print("=" * 40)
            print(f"\nGenerated in {(t1-t0):.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main() 