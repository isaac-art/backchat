import torch
import os
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer
from contextlib import nullcontext

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

if torch.cuda.is_available(): DEVICE = "cuda"
elif torch.backends.mps.is_available():  DEVICE = "mps"
else: DEVICE = "cpu"

if DEVICE == "cuda":  CTX = torch.autocast(DEVICE, dtype=torch.bfloat16) 
elif DEVICE == "mps": CTX = nullcontext()  # No autocast for MPS
else: CTX = torch.autocast("cpu", dtype=torch.bfloat16)

class DemoTokenizer:
    """Simple tokenizer for demo mode that splits text into words."""
    def __init__(self):
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
    
    def encode(self, s: str, bos: bool, eos: bool) -> list:
        tokens = list(range(3, len(s.split()) + 3))  # Start from 3 to avoid special tokens
        if bos: tokens = [self.bos_id] + tokens
        if eos: tokens = tokens + [self.eos_id]
        return tokens
    
    def decode(self, tokens: list) -> str:
        if not tokens: return ""
        tokens = [t for t in tokens if t not in [self.bos_id, self.eos_id, self.pad_id]]
        return " ".join(f"word_{t}" for t in tokens)

class DemoModel:
    """Simple model for demo mode that returns a predefined response."""
    def __init__(self): self.config = GPTConfig()
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, x): return torch.zeros((1, x.shape[1], self.config.vocab_size)), None


def load_model():
    if DEMO_MODE:
        print("Running in DEMO mode with mock model")
        return DemoModel()
    
    print(f"Loading chat model on {DEVICE}...")
    load_device = 'cpu' if DEVICE == 'mps' else DEVICE
    checkpoint = torch.load("out/checkpoints_chat/best_checkpoint_chat.pt", map_location=load_device)
    
    # Initialize model with chat-specific config
    model_config = GPTConfig(**checkpoint["model_args"])
    model = GPT(model_config)
    
    # Handle _orig_mod prefix in state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def get_tokenizer():
    if DEMO_MODE: return DemoTokenizer()
    return Tokenizer("data/tok8192_chat.model")  # Using chat-specific tokenizer

model = load_model()
tokenizer = get_tokenizer() 