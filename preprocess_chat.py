import json
import requests
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


# Constants
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DOLLY_FILE = DATA_DIR / "dolly.jsonl"
# Use the shared tokenizer model from preprocess_fine
TOKENIZER_MODEL = DATA_DIR / "tokenizer.model"


def download_dataset():
    """Download Dolly dataset if not exists"""
    if not DOLLY_FILE.exists():
        print("Downloading Dolly dataset...")
        response = requests.get(DOLLY_URL, stream=True)
        with open(DOLLY_FILE, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)


# SPECIAL_TOKENS = [
#     "<|endoftext|>",
#     "<|im_start|>",
#     "<|im_end|>",
#     "<|instruction|>",
#     "<|context|>",
#     "<|system|>",
#     "<|response|>",
#     "<|user|>",
#     "<|assistant|>"
# ]

def prepare_chat_text(example):
    """Prepare chat text from a single example"""
    # context = example.get('context', '').strip()
    instruction = example['instruction'].strip()
    response = example['response'].strip()
    
    full_text = f"<|instruction|> {instruction} <|response|> {response} <|endoftext|>"

    all_words = full_text.split()
    reversed_text = " ".join(all_words[::-1])
    return reversed_text


def check_tokenizer_exists() -> bool:
    """Check if the shared tokenizer model exists"""
    return TOKENIZER_MODEL.exists()


def process_shard(args: tuple) -> None:
    shard_id, examples = args
    from tokenizer import Tokenizer
    
    # Use the shared tokenizer model
    if not TOKENIZER_MODEL.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found at {TOKENIZER_MODEL}. "
            "Please run 'python preprocess_fine.py --vocab-size 8888' first to train the tokenizer."
        )
    
    tokenizer = Tokenizer(str(TOKENIZER_MODEL))
    all_tokens = []
    
    for example in tqdm(examples, position=shard_id):
        reversed_text = prepare_chat_text(example)
        tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    output_file = DATA_DIR / f"chat_tokens_{shard_id:02d}.bin"
    with open(output_file, "wb") as f:
        f.write(all_tokens.tobytes())


def pretokenize(num_workers: int = 4) -> None:
    """Pretokenize the dataset using the shared tokenizer"""
    if not check_tokenizer_exists():
        raise FileNotFoundError(
            f"Tokenizer model not found at {TOKENIZER_MODEL}. "
            "Please run 'python preprocess_fine.py --vocab-size 8888' first to train the tokenizer."
        )
    
    print("Loading dataset...")
    with open(DOLLY_FILE, "r") as f:
        examples = [json.loads(line) for line in f]
    
    # Split examples into shards
    shard_size = len(examples) // num_workers
    shards = [(i, examples[i*shard_size:(i+1)*shard_size]) for i in range(num_workers)]
    if len(examples) % num_workers:
        shards[-1] = (num_workers-1, examples[(num_workers-1)*shard_size:])
    
    print(f"Pretokenizing with {num_workers} workers using tokenizer from {TOKENIZER_MODEL}...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_shard, shards))


def prepare_dataset() -> None:
    """Prepare the chat dataset using the shared tokenizer"""
    print("Step 1: Downloading dataset...")
    download_dataset()
    
    print("\nStep 2: Checking for shared tokenizer...")
    if not check_tokenizer_exists():
        print(f"ERROR: Tokenizer model not found at {TOKENIZER_MODEL}")
        print("Please run 'python preprocess_fine.py --vocab-size 8888' first to train the tokenizer.")
        print("This ensures both datasets use the same vocabulary.")
        return
    
    print(f"Found tokenizer at {TOKENIZER_MODEL}")
    
    print("\nStep 3: Pretokenizing dataset...")
    pretokenize()
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process chat dataset")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset")
    
    # Pretokenize command
    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare-dataset", help="Run all dataset preparation steps")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_dataset()
    elif args.command == "pretokenize":
        pretokenize(args.num_workers)
    elif args.command == "prepare-dataset":
        prepare_dataset()
    else:
        parser.print_help() 