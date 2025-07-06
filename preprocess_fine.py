import os
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sentencepiece as spm
from functools import partial
from concurrent.futures import ProcessPoolExecutor


# Constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FINE_DIR = DATA_DIR / "fine"
FINE_DIR.mkdir(exist_ok=True)

def download_dataset(num_chunks: int = 100) -> None:
    """Download Fineweb dataset in chunks"""
    print("Downloading Fineweb-10BT dataset...")
    
    # Create checkpoint file
    checkpoint_file = FINE_DIR / "download_checkpoint.txt"
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
            chunk_path = FINE_DIR / f"chunk_{chunk_idx:05d}.txt"
            if not chunk_path.exists():  # Don't overwrite existing chunks
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_chunk))
            with open(checkpoint_file, 'w') as f:
                f.write(str(chunk_idx))
            current_chunk = []
            chunk_idx += 1

def train_vocab(vocab_size: int) -> None:
    """Train tokenizer with special tokens for instruction tuning"""
    prefix = DATA_DIR / "tokenizer"
    train_file = DATA_DIR / "fine_sample.txt"
    
    print("Preparing text for vocab training...")
    
    # Sample from chunks for vocab training
    chunk_files = sorted(list(FINE_DIR.glob("chunk_*.txt")))
    with open(train_file, "w", encoding="utf-8") as out:
        # Write special tokens first (excluding endoftext since it's a control symbol)
        special_tokens_no_eos = [
            "<|im_start|>",
            "<|im_end|>",
            "<|instruction|>",
            "<|response|>",
            "<|context|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>"
        ]
        for token in special_tokens_no_eos:
            out.write(token + "\n")
        
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
    tokenizer_model = DATA_DIR / "tokenizer.model"
    
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
    chunk_files = sorted(list(FINE_DIR.glob("chunk_*.txt")))
    
    print(f"Pretokenizing with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(partial(process_chunk, vocab_size=vocab_size), 
                    enumerate(chunk_files))

def prepare_dataset(vocab_size: int = 8888, num_chunks: int = 100) -> None:
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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset")
    download_parser.add_argument("--num-chunks", type=int, default=100, help="Number of chunks to download")

    # Train vocab command
    vocab_parser = subparsers.add_parser("train-vocab", help="Train tokenizer vocabulary")
    vocab_parser.add_argument("--vocab-size", type=int, default=8888, help="Size of vocabulary to train")

    # Pretokenize command
    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument("--vocab-size", type=int, default=8888, help="Vocabulary size (for tokenizer path)")
    pretok_parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")

    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare-dataset", help="Run all dataset preparation steps")
    prepare_parser.add_argument("--vocab-size", type=int, default=8888, help="Size of vocabulary to train")
    prepare_parser.add_argument("--num-chunks", type=int, default=100, help="Number of chunks to process")
    prepare_parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")

    args = parser.parse_args()

    if args.command == "download":
        download_dataset(args.num_chunks)
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size)
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size, args.num_workers)
    elif args.command == "prepare-dataset":
        prepare_dataset(args.vocab_size, args.num_chunks)
        pretokenize(args.vocab_size, args.num_workers)
    else:
        parser.print_help() 