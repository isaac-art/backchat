import os
import argparse
import json
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sentencepiece as spm

# Constants
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DOLLY_FILE = DATA_DIR / "dolly.jsonl"

def download_dataset():
    """Download Dolly dataset if not exists"""
    if not DOLLY_FILE.exists():
        print("Downloading Dolly dataset...")
        response = requests.get(DOLLY_URL, stream=True)
        with open(DOLLY_FILE, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)

def prepare_chat_text(example):
    """Prepare chat text from a single example"""
    context = example.get('context', '').strip()
    instruction = example['instruction'].strip()
    response = example['response'].strip()
    
    if context:
        full_text = f"Context: {context}\n\nInstruction: {instruction}\n\nResponse: {response}"
    else:
        full_text = f"Instruction: {instruction}\n\nResponse: {response}"
    
    # Reverse the entire sequence
    all_words = full_text.split()
    reversed_text = " ".join(all_words[::-1])
    return reversed_text

def train_vocab(vocab_size: int) -> None:
    prefix = DATA_DIR / f"tok{vocab_size}_chat"
    train_file = DATA_DIR / "chat_sample.txt"
    
    print("Preparing text for vocab training...")
    with open(DOLLY_FILE, "r") as f, open(train_file, "w", encoding="utf-8") as out:
        for line in tqdm(f):
            example = json.loads(line)
            reversed_text = prepare_chat_text(example)
            out.write(reversed_text + "\n")
    
    print(f"\nTraining tokenizer with vocab size {vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r"\342\201\207 ",
        normalization_rule_name="identity",
    )

def process_shard(args: tuple, vocab_size: int) -> None:
    shard_id, examples = args
    from tokenizer import Tokenizer
    
    tokenizer_model = DATA_DIR / f"tok{vocab_size}_chat.model"
    tokenizer = Tokenizer(str(tokenizer_model))
    all_tokens = []
    
    for example in tqdm(examples, position=shard_id):
        reversed_text = prepare_chat_text(example)
        tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    output_file = DATA_DIR / f"chat_tokens_{shard_id:02d}.bin"
    with open(output_file, "wb") as f:
        f.write(all_tokens.tobytes())

def pretokenize(vocab_size: int, num_workers: int = 4) -> None:
    print("Loading dataset...")
    with open(DOLLY_FILE, "r") as f:
        examples = [json.loads(line) for line in f]
    
    # Split examples into shards
    shard_size = len(examples) // num_workers
    shards = [(i, examples[i*shard_size:(i+1)*shard_size]) for i in range(num_workers)]
    if len(examples) % num_workers:
        shards[-1] = (num_workers-1, examples[(num_workers-1)*shard_size:])
    
    print(f"Pretokenizing with {num_workers} workers...")
    func = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(func, shards))

def prepare_dataset(vocab_size: int) -> None:
    print("Step 1: Downloading dataset...")
    download_dataset()
    
    print("\nStep 2: Training vocabulary...")
    train_vocab(vocab_size)
    
    print("\nStep 3: Pretokenizing dataset...")
    pretokenize(vocab_size)
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process chat dataset")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset")
    
    # Train vocab command
    vocab_parser = subparsers.add_parser("train-vocab", help="Train vocabulary")
    vocab_parser.add_argument("--vocab-size", type=int, required=True, help="Size of vocabulary to train")
    
    # Pretokenize command
    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size to use for tokenization")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare-dataset", help="Run all dataset preparation steps")
    prepare_parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size for training and tokenization")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_dataset()
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size)
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size)
    elif args.command == "prepare-dataset":
        prepare_dataset(args.vocab_size)
    else:
        parser.print_help() 