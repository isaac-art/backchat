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
    def iter_batches(batch_size, max_seq_len, split, device="cpu", num_workers=0, dataset="tiny"):
        """Iterator for batches of data"""
        data_dir = Path("data")
        
        # Determine which dataset to use
        if dataset == "tiny":
            pattern = str(data_dir / "*.bin")
        elif dataset == "chat":
            pattern = str(data_dir / "chat_tokens_*.bin")
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")
        
        # Get all data files
        data_files = sorted(glob.glob(pattern))
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern: {pattern}")
        
        # Split into train/val
        split_idx = int(0.9 * len(data_files))
        train_files = data_files[:split_idx]
        val_files = data_files[split_idx:]
        
        # Select files based on split
        files = train_files if split == "train" else val_files
        if not files:
            raise ValueError(f"No files found for split: {split}")
        
        # Create dataset and dataloader
        ds = TokenDataset(files, max_seq_len)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers
        )
        
        while True:
            for batch in dl:
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                yield x, y

class TokenDataset(Dataset):
    def __init__(self, data_files, max_seq_len):
        self.data_files = data_files
        self.max_seq_len = max_seq_len
        
        # Load all data into memory
        data = []
        for file in data_files:
            with open(file, 'rb') as f:
                data.extend(np.frombuffer(f.read(), dtype=np.uint16))
        self.data = data
        
        # Calculate number of sequences
        self.num_sequences = (len(self.data) - 1) // self.max_seq_len
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get sequence starting position
        start_idx = idx * self.max_seq_len
        
        # Extract sequence
        chunk = self.data[start_idx:start_idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y
