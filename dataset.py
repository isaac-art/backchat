import glob
import torch
import random
import numpy as np
from pathlib import Path
import torch.distributed as dist
from typing import Iterator, Tuple

class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, max_seq_len: int, bin_dir: Path):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.bin_dir = bin_dir

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        shard_filenames = sorted(glob.glob(str(self.bin_dir / "*.bin")))
        # print(f"[DEBUG][PreTokDataset] Split: {self.split}, Found {len(shard_filenames)} .bin files: {shard_filenames}")
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        # print(f"[DEBUG][PreTokDataset] Using files for split '{self.split}': {shard_filenames}")
        rng = random.Random(seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # print(f"[DEBUG][PreTokDataset] Loading shard: {shard}")
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                # print(f"[DEBUG][PreTokDataset] Shard {shard} has {num_batches} batches (data len: {len(data)}, max_seq_len: {self.max_seq_len})")
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    # print(f"[DEBUG][PreTokDataset] Yielding batch idx {idx} from shard {shard}")
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # print(f"[DEBUG][Task.iter_batches] Creating PreTokDataset with args: {dataset_kwargs}")
        ds = PreTokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers
        )
        # print(f"[DEBUG][Task.iter_batches] DataLoader created, batch_size={batch_size}, num_workers={num_workers}")
        for x, y in dl:
            # print(f"[DEBUG][Task.iter_batches] Yielding batch from DataLoader")
            x = x.to(device)
            y = y.to(device)
            yield x, y
