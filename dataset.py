import torch
import numpy as np
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple, List
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dataset configurations
DATASETS = {
    "tiny": {
        "name": "TinyStories",
        "dir": Path("data/TinyStories_all_data"),
    },
    "owt2": {
        "name": "OpenWebText2",
        "dir": Path("data/openwebtext2"),
    },
    "chat": {
        "name": "Chat",
        "dir": Path("data/chat_tokens"),
    },
    "fine10": {
        "name": "Fine10",
        "dir": Path("data/fine10"),
    }
}

class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, max_seq_len: int, dataset: str = "tiny"):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.data_files: List[str] = []
        self._load_data_files()

    def _load_data_files(self) -> None:
        """Load all available preprocessed binary files."""
        config = DATASETS[self.dataset]
        bin_dir = config['dir']
        pattern = "*.bin"  # All preprocessed binary files
        self.data_files = sorted(glob.glob(str(bin_dir / pattern)))
        
        if not self.data_files:
            raise RuntimeError(f"No preprocessed files found in {bin_dir}. Please run preprocessing first.")
        
        # For validation, use a fixed subset of the data
        if self.split == "val":
            random.seed(42)  # Fixed seed for validation
            if self.dataset == "tiny":
                # For TinyStories, use first shard for validation as before
                self.data_files = self.data_files[:1]
            else:
                # For OpenWebText2, use 1% of shards for validation
                num_val_files = max(1, len(self.data_files) // 100)
                self.data_files = random.sample(self.data_files, num_val_files)
        elif self.dataset == "tiny" and self.split == "train":
            # For TinyStories train, use all except first shard
            self.data_files = self.data_files[1:]
        
        print(f"Loaded {len(self.data_files)} files for {self.split} split of {config['name']}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Divide files among workers
        files_per_worker = max(len(self.data_files) // num_workers, 1)
        start_idx = worker_id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.data_files)
        worker_files = self.data_files[start_idx:end_idx]

        if not worker_files:
            raise RuntimeError(f"No files assigned to worker {worker_id}")

        rng = random.Random(42 + worker_id)  # Different seed for each worker
        
        while True:
            rng.shuffle(worker_files)
            for file_path in worker_files:
                try:
                    data = np.memmap(file_path, dtype=np.uint16, mode="r")
                    
                    # Calculate number of complete sequences
                    num_seq = len(data) // self.max_seq_len
                    if num_seq == 0:
                        continue
                        
                    # Generate random indices for sequences
                    indices = list(range(num_seq))
                    rng.shuffle(indices)
                    
                    for idx in indices:
                        start_idx = idx * self.max_seq_len
                        end_idx = start_idx + self.max_seq_len
                        
                        if end_idx <= len(data):
                            chunk = torch.from_numpy(data[start_idx:end_idx].astype(np.int64))
                            x = chunk[:-1]
                            y = chunk[1:]
                            yield x, y
                            
                except (ValueError, OSError) as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue


class Task:
    @staticmethod
    def iter_batches(batch_size, max_seq_len, device, dataset="tiny", split="train", num_workers=0):
        paths = {
            "tiny": DATASETS["tiny"]["dir"],
            "owt2": DATASETS["owt2"]["dir"],
            "fine10": DATASETS["fine10"]["dir"],
            "chat": DATASETS["chat"]["dir"],
        }
        
        if dataset not in paths:
            raise ValueError(f"Unknown dataset type: {dataset}")
            
        data_dir = paths[dataset]
        
        # Get all .bin files for the dataset
        bin_files = sorted(glob.glob(str(data_dir / "*.bin")))
        if not bin_files:
            raise ValueError(f"No .bin files found in {data_dir}")
        
        # Split into train/val
        num_val = max(1, int(len(bin_files) * 0.1))  # 10% for validation
        train_files = bin_files[:-num_val]
        val_files = bin_files[-num_val:]
        
        # Select the appropriate split
        files = train_files if split == 'train' else val_files
        if not files:
            raise ValueError(f"No files found for {split} split")
            
        # Create dataset and dataloader
        dataset = BinaryDataset(files, max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            pin_memory=True,
            num_workers=num_workers
        )
        
        while True:
            for batch in loader:
                X = batch[:, :-1].to(device)  # Input
                Y = batch[:, 1:].to(device)   # Target
                yield X, Y

class BinaryDataset(Dataset):
    def __init__(self, files, max_seq_len):
        self.files = files
        self.max_seq_len = max_seq_len
        self.vocab_size = None  # Will be set when loading data
        
        # Load and concatenate all data
        data_chunks = []
        for file in files:
            with open(file, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint16)
                data_chunks.append(data)
        self.data = np.concatenate(data_chunks)
        
        # Calculate number of sequences
        self.num_sequences = (len(self.data) - 1) // max_seq_len
        
    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        # Get sequence starting at idx * max_seq_len
        start_idx = idx * self.max_seq_len
        chunk = self.data[start_idx:start_idx + self.max_seq_len + 1]
        # Pad if necessary (should rarely happen)
        if len(chunk) < self.max_seq_len + 1:
            chunk = np.pad(chunk, (0, self.max_seq_len + 1 - len(chunk)))
        return torch.from_numpy(chunk.astype(np.int64))
