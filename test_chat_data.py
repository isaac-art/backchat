import torch
from dataset import Task
from pathlib import Path
from tokenizer import Tokenizer, TOKENIZER_MODEL

# Settings (match your training config)
BATCH_SIZE = 1
MAX_SEQ_LEN = 512  # adjust if needed to match your model
BIN_DIR = Path("data")
DEVICE = "cpu"  # use cpu for testing


def main():
    # Load tokenizer
    tokenizer = Tokenizer(TOKENIZER_MODEL)

    # Create batch iterator (like in training)
    batch_iter = Task.iter_batches(
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        bin_dir=BIN_DIR,
        device=DEVICE,
    )

    # Get one batch
    x, y = next(batch_iter)
    x = x[0].tolist()  # take first sequence in batch
    y = y[0].tolist()

    print("First input token sequence:", x)
    print("First target token sequence:", y)

    # Decode (untokenize) the input sequence
    decoded = tokenizer.decode(x)
    print("\nDecoded input sequence:")
    print(decoded)

    # Optionally decode the target sequence
    decoded_target = tokenizer.decode(y)
    print("\nDecoded target sequence:")
    print(decoded_target)

if __name__ == "__main__":
    main() 