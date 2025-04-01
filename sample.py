import torch
import argparse
from tokenizer import Tokenizer
from model import GPT
from config import GPTConfig
from contextlib import nullcontext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--dataset", type=str, default="tiny", choices=["tiny", "owt2"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    # Update default paths to use dataset-specific names
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.tokenizer_path is None:
        args.tokenizer_path = f"data/tok50257_{args.dataset}.model" if args.dataset == "owt2" else f"data/tok4096_{args.dataset}.model"
    if args.ckpt_path is None:
        args.ckpt_path = f"out/checkpoints_{args.dataset}/best_checkpoint_{args.dataset}.pt"
    
    return args


def setup_device(args):
    torch.manual_seed(args.seed)
    
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast(args.device, dtype=torch.bfloat16)
    elif args.device == "mps":
        return nullcontext()  # No autocast for MPS
    else:
        return torch.autocast("cpu", dtype=torch.bfloat16)


def load_model(args):
    # For MPS, load to CPU first then transfer
    load_device = 'cpu' if args.device == 'mps' else args.device
    checkpoint = torch.load(args.ckpt_path, map_location=load_device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    # For MPS, move to device after model is fully initialized
    model.to(args.device)
    if args.compile and args.device != "mps":  # Don't compile for MPS
        model = torch.compile(model)
    return model


def main():
    args = parse_args()
    
    # Convert MPS dtype if needed
    if args.device == "mps" and args.dtype == "bfloat16":
        args.dtype = "float16"  # MPS doesn't support bfloat16
    
    ctx = setup_device(args)
    model = load_model(args)

    enc = Tokenizer(args.tokenizer_path)
    encode = lambda s: enc.encode(s, bos=True, eos=False)
    decode = lambda l: enc.decode(l)

    x = torch.tensor(
        encode(args.prompt), dtype=torch.long, device=args.device
    ).unsqueeze(0)

    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(
                    x,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                )
                print(decode(y[0].tolist()))
                print("------------------")


if __name__ == "__main__":
    main()