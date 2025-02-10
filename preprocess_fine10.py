import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sentencepiece as spm
import datasets
from typing import List, Dict

# Constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FINE10_DIR = DATA_DIR / "fine10"
FINE10_DIR.mkdir(exist_ok=True)

# Special tokens for instruction tuning
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|instruction|>",
    "<|response|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>"
]

def download_dataset(num_chunks: int = 100) -> None:
    """Download Fineweb dataset in chunks"""
    print("Downloading Fineweb-10BT dataset...")
    
    # Create checkpoint file
    checkpoint_file = FINE10_DIR / "download_checkpoint.txt"
    start_chunk = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            start_chunk = int(f.read().strip())
        print(f"Resuming from chunk {start_chunk}")
    
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",  # Using the 10BT sample
        streaming=True,
        split="train"
    )
    
    chunk_size = 10000
    current_chunk = []
    chunk_idx = start_chunk
    
    for item in tqdm(ds, desc="Processing chunks"):
        if chunk_idx >= num_chunks:
            break
            
        if item.get('text'):
            current_chunk.append(item['text'])
            
        if len(current_chunk) >= chunk_size:
            chunk_path = FINE10_DIR / f"chunk_{chunk_idx:05d}.txt"
            if not chunk_path.exists():  # Don't overwrite existing chunks
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_chunk))
            with open(checkpoint_file, 'w') as f:
                f.write(str(chunk_idx))
            current_chunk = []
            chunk_idx += 1

def train_vocab(vocab_size: int) -> None:
    """Train tokenizer with special tokens for instruction tuning"""
    prefix = DATA_DIR / f"tok{vocab_size}_fine10"
    train_file = DATA_DIR / "fine10_sample.txt"
    
    print("Preparing text for vocab training...")
    
    # Sample from chunks for vocab training
    chunk_files = sorted(list(FINE10_DIR.glob("chunk_*.txt")))
    with open(train_file, "w", encoding="utf-8") as out:
        # Write special tokens first (excluding endoftext since it's a control symbol)
        special_tokens_no_eos = [
            "<|im_start|>",
            "<|im_end|>",
            "<|instruction|>",
            "<|response|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>"
        ]
        for token in special_tokens_no_eos:
            out.write(token + "\n")
        
        # Sample from data chunks
        for chunk_file in tqdm(chunk_files[:5]):  # Use first 5 chunks
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line.strip() + "\n")
    
    print(f"\nTraining tokenizer with vocab size {vocab_size}...")
    
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size - len(special_tokens_no_eos) - 1,  # -1 for endoftext
        user_defined_symbols=special_tokens_no_eos,
        control_symbols=["<|endoftext|>"],
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=3,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        normalization_rule_name="identity",
    )

def process_chunk(args: tuple, vocab_size: int) -> None:
    """Process and tokenize a single chunk"""
    chunk_id, chunk_file = args
    tokenizer_model = DATA_DIR / f"tok{vocab_size}_fine10.model"
    
    # Skip if output already exists
    output_file = chunk_file.with_suffix('.bin')
    if output_file.exists():
        return
        
    from tokenizer import Tokenizer
    tokenizer = Tokenizer(str(tokenizer_model))
    
    all_tokens = []
    with open(chunk_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, position=chunk_id, desc=f"Processing {chunk_file.name}"):
            text = line.strip()
            if text:
                # Reverse the text for training
                words = text.split()
                reversed_text = " ".join(words[::-1])
                tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
                all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open(output_file, "wb") as f:
        f.write(all_tokens.tobytes())

def pretokenize(vocab_size: int, num_workers: int = 4) -> None:
    """Pretokenize all chunks"""
    chunk_files = sorted(list(FINE10_DIR.glob("chunk_*.txt")))
    
    print(f"Pretokenizing with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(partial(process_chunk, vocab_size=vocab_size), 
                    enumerate(chunk_files))

def prepare_dataset(vocab_size: int = 8000, num_chunks: int = 100) -> None:
    """Run all preparation steps"""
    print("Step 1: Downloading dataset...")
    download_dataset(num_chunks)
    
    print("\nStep 2: Training vocabulary...")
    train_vocab(vocab_size)
    
    print("\nStep 3: Pretokenizing dataset...")
    pretokenize(vocab_size)
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Fineweb-10BT dataset")
    parser.add_argument("--vocab-size", type=int, default=8000,
                      help="Size of vocabulary to train")
    parser.add_argument("--num-chunks", type=int, default=100,
                      help="Number of chunks to process")
    
    args = parser.parse_args()
    prepare_dataset(args.vocab_size, args.num_chunks) 