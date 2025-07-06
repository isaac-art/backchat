import torch
import argparse
from tokenizer import Tokenizer
from model import GPT
from config import GPTConfig
from contextlib import nullcontext


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained GPT model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt to start generation")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer model")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--min_p", type=float, default=0.05, help="Minimum probability for min_p sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (bfloat16, float16, float32)")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--mode", type=str, default="fine", choices=["fine", "chat"], help="Model type: fine or chat")
    args = parser.parse_args()

    # Set default paths if not provided
    if args.tokenizer_path is None:
        args.tokenizer_path = "data/tokenizer.model"
    if args.ckpt_path is None:
        ckpt_name = "best_checkpoint_fine.pt" if args.mode == "fine" else "best_checkpoint_chat.pt"
        ckpt_dir = "out/checkpoints_fine" if args.mode == "fine" else "out/checkpoints_chat"
        args.ckpt_path = f"{ckpt_dir}/{ckpt_name}"
    return args


def setup_device(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast(args.device, dtype=getattr(torch, args.dtype))
    elif args.device == "mps":
        return nullcontext()
    else:
        return torch.autocast("cpu", dtype=getattr(torch, args.dtype))


def load_model(args):
    load_device = 'cpu' if args.device == 'mps' else args.device
    checkpoint = torch.load(args.ckpt_path, map_location=load_device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    if args.compile and args.device != "mps":
        model = torch.compile(model)
    return model


def main():
    args = parse_args()
    if args.device == "mps" and args.dtype == "bfloat16":
        args.dtype = "float16"
    ctx = setup_device(args)
    model = load_model(args)
    enc = Tokenizer(args.tokenizer_path)
    encode = lambda s: enc.encode(s, bos=True, eos=False)
    decode = lambda l: enc.decode(l)
    x = torch.tensor(encode(args.prompt), dtype=torch.long, device=args.device).unsqueeze(0)
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


# # Sample from the fineweb model
# uv run sample.py --prompt "The history of AI" --mode fine

# # Sample from the chat-finetuned model
# uv run sample.py --prompt "How do I train a neural network?" --mode chat
