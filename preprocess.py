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
import zstandard as zstd
import jsonlines
import tarfile
from typing import Iterator, List
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
    "owt2": {
        "name": "OpenWebText2",
        "url": "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar",
        "dir": DATA_CACHE_DIR / "openwebtext2",
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
    else:  # owt2
        data_filename = DATA_CACHE_DIR / "openwebtext2.tar"
        config['dir'].mkdir(exist_ok=True)
        
        if not data_filename.exists():
            print(f"Downloading {config['name']} dataset...")
            download_file(config['url'], str(data_filename))
            
        if not any(config['dir'].glob("*.jsonl.zst")):
            print(f"Extracting {config['name']} dataset...")
            with tarfile.open(data_filename) as tar:
                tar.extractall(path=config['dir'])

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
        else:  # owt2
            reader = Reader()
            shard_filenames = sorted(glob.glob(str(config['dir'] / "*.jsonl.zst")))
            count = 0
            for shard in shard_filenames[:2]:  # Use first 2 shards for vocab training
                for text, _ in reader.read_jsonl(shard):
                    f.write(text + "\n")
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
    else:  # owt2
        reader = Reader()
        for text, _ in tqdm(reader.read_jsonl(shard), position=shard_id, desc=f"Processing {Path(shard).name}"):
            reversed_text = reverse_sentence(text)
            tokens = tokenizer.encode(reversed_text, bos=True, eos=True)
            all_tokens.extend(tokens)

            if len(all_tokens) > 1_000_000:
                tokens_array = np.array(all_tokens, dtype=np.uint16)
                tokenized_filename = str(shard).replace(".jsonl.zst", f"_{len(all_tokens)}.bin")
                with open(tokenized_filename, "wb") as f:
                    f.write(tokens_array.tobytes())
                all_tokens = []

        if all_tokens:
            tokens_array = np.array(all_tokens, dtype=np.uint16)
            tokenized_filename = str(shard).replace(".jsonl.zst", f"_{len(all_tokens)}.bin")
            with open(tokenized_filename, "wb") as f:
                f.write(tokens_array.tobytes())

def pretokenize(vocab_size: int, dataset: str = "tiny") -> None:
    config = DATASETS[dataset]
    pattern = "*.json" if dataset == "tiny" else "*.jsonl.zst"
    shard_filenames = sorted(glob.glob(str(config['dir'] / pattern)))

    func = partial(process_shard, vocab_size=vocab_size, dataset=dataset)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))

def prepare_dataset(vocab_size: int, dataset: str = "tiny") -> None:
    config = DATASETS[dataset]
    print(f"Step 1: Downloading {config['name']} dataset...")
    download(dataset)
    
    if dataset == "owt2":
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
    parser.add_argument(
        "--dataset",
        choices=["tiny", "owt2"],
        default="tiny",
        help="Dataset to process (default: tiny)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    download_parser = subparsers.add_parser(
        "download", help="Download dataset"
    )

    vocab_parser = subparsers.add_parser("train-vocab", help="Train vocabulary")
    vocab_parser.add_argument(
        "--vocab-size", type=int, required=True, help="Size of vocabulary to train"
    )

    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size to use for tokenization",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Run all dataset preparation steps sequentially"
    )
    prepare_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for training and tokenization",
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
