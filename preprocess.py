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
import glob
import math

from tokenizer import Tokenizer
from utils.archiver import Reader

DATA_CACHE_DIR = Path("data")
DATA_CACHE_DIR.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    "tiny": {
        "name": "TinyStories",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",
        "dir": DATA_CACHE_DIR / "TinyStories_all_data",
    },
    "fineweb": {
        "name": "Fineweb",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        "dir": DATA_CACHE_DIR / "fineweb",
    }
}

def download_file(url: str, filename: str, chunk_size: int = 1024) -> None:
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)

def download(dataset: str = "tiny") -> None:
    config = DATASETS[dataset]
    if dataset == "tiny":
        data_filename = DATA_CACHE_DIR / "TinyStories_all_data.tar.gz"
        if not data_filename.exists():
            print(f"Downloading {config['name']} dataset...")
            download_file(config['url'], str(data_filename))

        if not config['dir'].exists():
            config['dir'].mkdir(exist_ok=True)
            print(f"Extracting {config['name']} dataset...")
            os.system(f"tar --no-same-owner -xvf {data_filename} -C {config['dir']}")
    else:  # fineweb
        try:
            import datasets
            print(f"Downloading {config['name']} dataset...")
            config['dir'].mkdir(exist_ok=True)
            
            # Add checkpoint tracking
            checkpoint_file = config['dir'] / "checkpoint.txt"
            start_chunk = 0
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    start_chunk = int(f.read().strip())
                print(f"Resuming from chunk {start_chunk}")
            
            ds = datasets.load_dataset(
                "HuggingFaceFW/fineweb",
                streaming=True,
                split="train"
            )
            
            chunk_size = 10000
            current_chunk = []
            chunk_idx = start_chunk
            
            # Add progress tracking
            processed = 0
            print(f"Processing Fineweb dataset from chunk {chunk_idx}...")
            
            for item in tqdm(ds, initial=chunk_idx * chunk_size):
                if processed < chunk_idx * chunk_size:
                    processed += 1
                    continue
                    
                if item.get('text'):
                    current_chunk.append(item['text'])
                    
                if len(current_chunk) >= chunk_size:
                    chunk_path = config['dir'] / f"chunk_{chunk_idx:05d}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_chunk))
                    with open(checkpoint_file, 'w') as f:
                        f.write(str(chunk_idx))
                    current_chunk = []
                    chunk_idx += 1
                    
                # Optional: Add early stopping for testing
                # if chunk_idx >= 10:  # Only process 10 chunks
                #     break
            
            # Save any remaining examples
            if current_chunk:
                chunk_path = config['dir'] / f"chunk_{chunk_idx:05d}.txt"
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_chunk))
                    
        except Exception as e:
            print(f"Error downloading Fineweb dataset: {e}")
            raise

def get_dataset_stats(dataset_dir: Path) -> tuple[int, float]:
    """Get document count and total text size in GB for the dataset."""
    document_count = 0
    total_text_size = 0
    reader = Reader()
    
    files = glob.glob(str(dataset_dir / "*.jsonl.zst"))
    for file_path in tqdm(files, desc="Calculating dataset stats"):
        for document, _ in reader.read_jsonl(file_path, get_meta=True):
            document_count += 1
            total_text_size += len(document)
    
    return document_count, total_text_size / math.pow(10, 9)  # Convert to GB

def train_vocab(vocab_size: int, dataset: str = "tiny") -> None:
    prefix = DATA_CACHE_DIR / f"tok{vocab_size}_{dataset}"
    tiny_file = DATA_CACHE_DIR / f"sample_{dataset}.txt"
    config = DATASETS[dataset]

    with open(tiny_file, "w", encoding="utf-8") as f:
        if dataset == "tiny":
            shard_filenames = sorted(glob.glob(str(config['dir'] / "*.json")))
            for shard in shard_filenames[:10]:
                with open(shard, "r") as g:
                    data = json.load(g)
                for example in data:
                    f.write(example["story"].strip() + "\n")
        else:  # fineweb
            chunk_files = sorted(glob.glob(str(config['dir'] / "chunk_*.txt")))
            count = 0
            for chunk_file in chunk_files[:5]:  # Use first 5 chunks for vocab training
                with open(chunk_file, 'r', encoding='utf-8') as g:
                    for line in g:
                        f.write(line.strip() + "\n")
                        count += 1
                        if count >= 100000:  # Sample 100k examples for vocab training
                            break
                if count >= 100000:
                    break

    print(f"\nTraining {dataset} tokenizer with vocab size {vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=str(tiny_file),
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

def reverse_sentence(text: str) -> str:
    """Reverse the order of words in a sentence while maintaining word integrity."""
    return " ".join(text.strip().split()[::-1])

def process_shard(args: tuple, vocab_size: int, dataset: str = "tiny") -> None:
    shard_id, shard = args
    tokenizer_model = DATA_CACHE_DIR / f"tok{vocab_size}_{dataset}.model"
    tokenizer = Tokenizer(str(tokenizer_model))
    all_tokens = []

    if dataset == "tiny":
        with open(shard, "r") as f:
            data = json.load(f)
        for example in tqdm(data, position=shard_id):
            text = example["story"].strip()
            reversed_text = reverse_sentence(text)
            tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
            all_tokens.extend(tokens)

        all_tokens = np.array(all_tokens, dtype=np.uint16)
        tokenized_filename = str(shard).replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
    else:  # fineweb
        with open(shard, 'r', encoding='utf-8') as f:
            for line in tqdm(f, position=shard_id, desc=f"Processing {Path(shard).name}"):
                text = line.strip()
                if text:  # Skip empty lines
                    reversed_text = reverse_sentence(text)
                    tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
                    all_tokens.extend(tokens)

                if len(all_tokens) > 1_000_000:
                    tokens_array = np.array(all_tokens, dtype=np.uint16)
                    tokenized_filename = str(shard).replace(".txt", f"_{len(all_tokens)}.bin")
                    with open(tokenized_filename, "wb") as f:
                        f.write(tokens_array.tobytes())
                    all_tokens = []

        if all_tokens:
            tokens_array = np.array(all_tokens, dtype=np.uint16)
            tokenized_filename = str(shard).replace(".txt", f"_{len(all_tokens)}.bin")
            with open(tokenized_filename, "wb") as f:
                f.write(tokens_array.tobytes())

def pretokenize(vocab_size: int, dataset: str = "tiny") -> None:
    config = DATASETS[dataset]
    pattern = "*.json" if dataset == "tiny" else "chunk_*.txt"
    shard_filenames = sorted(glob.glob(str(config['dir'] / pattern)))

    func = partial(process_shard, vocab_size=vocab_size, dataset=dataset)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))

def prepare_dataset(vocab_size: int, dataset: str = "tiny") -> None:
    config = DATASETS[dataset]
    print(f"Step 1: Downloading {config['name']} dataset...")
    download(dataset)
    
    if dataset == "fineweb":
        print("\nCalculating dataset statistics...")
        doc_count, size_gb = get_dataset_stats(config['dir'])
        print(f"Total documents: {doc_count:,}")
        print(f"Total uncompressed text size: {size_gb:.2f} GB")
    
    print("\nStep 2: Training vocabulary...")
    train_vocab(vocab_size, dataset)
    print("\nStep 3: Pretokenizing dataset...")
    pretokenize(vocab_size, dataset)
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download dataset"
    )
    download_parser.add_argument(
        "--dataset",
        choices=["tiny", "fineweb"],
        default="tiny",
        help="Dataset to process (default: tiny)"
    )
    download_parser.add_argument(
        "--max-chunks",
        type=int,
        help="Maximum number of chunks to process (for testing)",
        default=None
    )

    # Train vocab command
    vocab_parser = subparsers.add_parser("train-vocab", help="Train vocabulary")
    vocab_parser.add_argument(
        "--vocab-size", 
        type=int, 
        required=True, 
        help="Size of vocabulary to train"
    )
    vocab_parser.add_argument(
        "--dataset",
        choices=["tiny", "fineweb"],
        default="tiny",
        help="Dataset to process (default: tiny)"
    )

    # Pretokenize command
    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size to use for tokenization",
    )
    pretok_parser.add_argument(
        "--dataset",
        choices=["tiny", "fineweb"],
        default="tiny",
        help="Dataset to process (default: tiny)"
    )

    # Prepare dataset command
    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Run all dataset preparation steps sequentially"
    )
    prepare_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for training and tokenization",
    )
    prepare_parser.add_argument(
        "--dataset",
        choices=["tiny", "fineweb"],
        default="tiny",
        help="Dataset to process (default: tiny)"
    )

    args = parser.parse_args()

    if args.command == "download":
        download(args.dataset)
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size, args.dataset)
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size, args.dataset)
    elif args.command == "prepare-dataset":
        prepare_dataset(args.vocab_size, args.dataset)
    else:
        parser.print_help()

