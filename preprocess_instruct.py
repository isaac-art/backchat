import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import datasets
from tokenizer import Tokenizer

# Constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INSTRUCT_DIR = DATA_DIR / "instruct"
INSTRUCT_DIR.mkdir(exist_ok=True)

def format_conversation(instruction: str, input_text: str, output: str) -> str:
    """
    Format conversation for backwards generation.
    For our backwards model:
    1. Response comes first (in reverse) - this is what we'll provide
    2. Then instruction (in reverse) - this is what the model will learn to generate
    3. We use special tokens to mark boundaries
    """
    # Clean and combine instruction and input
    instruction = instruction.strip()
    input_text = input_text.strip() if input_text else ""
    full_instruction = f"{instruction} {input_text}".strip()
    
    # Clean output
    output = output.strip()
    
    # For backwards generation:
    # 1. Split output into words and reverse
    output_words = output.split()
    reversed_output = " ".join(output_words[::-1])
    
    # 2. Split instruction into words and reverse
    instruction_words = full_instruction.split()
    reversed_instruction = " ".join(instruction_words[::-1])
    
    # Format with special tokens
    # Note: In training data, we include both response and instruction
    # But during generation, we'll only provide response and model generates instruction
    formatted = f"<|im_start|><|response|>{reversed_output}<|im_end|>"
    formatted += f"<|im_start|><|instruction|>{reversed_instruction}<|im_end|>"
    
    return formatted

def download_and_preprocess_dataset():
    """Download and preprocess Open Instruct V1 dataset"""
    print("Downloading Open Instruct V1 dataset...")
    
    ds = datasets.load_dataset("hakurei/open-instruct-v1", split="train")
    
    # Process and save in chunks
    chunk_size = 10000
    current_chunk = []
    chunk_idx = 0
    
    print("Processing dataset...")
    for item in tqdm(ds):
        # Extract fields
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # Format conversation
        formatted = format_conversation(instruction, input_text, output)
        current_chunk.append(formatted)
        
        if len(current_chunk) >= chunk_size:
            chunk_path = INSTRUCT_DIR / f"chunk_{chunk_idx:05d}.txt"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_chunk))
            current_chunk = []
            chunk_idx += 1
    
    # Save any remaining items
    if current_chunk:
        chunk_path = INSTRUCT_DIR / f"chunk_{chunk_idx:05d}.txt"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(current_chunk))

def process_chunk(args: tuple, vocab_size: int = 8888) -> None:
    """Process and tokenize a single chunk using our pre-trained tokenizer"""
    chunk_id, chunk_file = args
    
    # Use the tokenizer we trained on Fineweb
    tokenizer_model = DATA_DIR / f"tok{vocab_size}_xcoax.model"
    if not tokenizer_model.exists():
        raise ValueError(f"Tokenizer not found at {tokenizer_model}. Please train on Fineweb first.")
    
    tokenizer = Tokenizer(str(tokenizer_model))
    
    # Skip if output already exists
    output_file = chunk_file.with_suffix('.bin')
    if output_file.exists():
        return
    
    all_tokens = []
    with open(chunk_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, position=chunk_id, desc=f"Processing {chunk_file.name}"):
            text = line.strip()
            if text:
                # Note: No need to reverse text here as it's already formatted correctly
                # Just encode with BOS/EOS tokens
                tokens = tokenizer.encode(text, bos=True, eos=True)
                all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open(output_file, "wb") as f:
        f.write(all_tokens.tobytes())

def pretokenize(vocab_size: int = 8888, num_workers: int = 8) -> None:
    """Pretokenize all chunks"""
    chunk_files = sorted(list(INSTRUCT_DIR.glob("chunk_*.txt")))
    
    print(f"Pretokenizing with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(partial(process_chunk, vocab_size=vocab_size), 
                    enumerate(chunk_files))

def prepare_dataset(vocab_size: int = 8888) -> None:
    """Run all preparation steps"""
    print("Step 1: Downloading and preprocessing dataset...")
    download_and_preprocess_dataset()
    
    print("\nStep 2: Tokenizing dataset...")
    pretokenize(vocab_size)
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Open Instruct V1 dataset")
    parser.add_argument("--vocab-size", type=int, default=8888,
                      help="Size of vocabulary (must match Fineweb tokenizer)")
    
    args = parser.parse_args()
    prepare_dataset(args.vocab_size)

# Example of how the formatting works:
"""
Original data:
{
    'instruction': 'Identity the odd one out.',
    'input': 'Twitter, Instagram, Telegram',
    'output': 'Telegram'
}

After formatting (before tokenization):
<|im_start|><|response|>Telegram<|im_end|><|im_start|><|instruction|>Twitter Instagram, Telegram out odd the Identity<|im_end|>

This way:
1. We provide the response first (backwards)
2. Model learns to generate the instruction (backwards)
3. During inference:
   - We give: <|im_start|><|response|>Telegram<|im_end|>
   - Model generates: <|im_start|><|instruction|>Twitter Instagram, Telegram out odd the Identity<|im_end|>
   - We un-reverse the instruction to get: "Identity the odd one out. Twitter, Instagram, Telegram"
""" 