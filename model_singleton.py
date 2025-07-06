import torch
import os
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer
from contextlib import nullcontext

DEMO_MODE = False

if torch.cuda.is_available(): DEVICE = "cuda"
elif torch.backends.mps.is_available():  DEVICE = "mps"
else: DEVICE = "cpu"

if DEVICE == "cuda":  CTX = torch.autocast(DEVICE, dtype=torch.bfloat16) 
elif DEVICE == "mps": CTX = nullcontext()  # No autocast for MPS
else: CTX = torch.autocast("cpu", dtype=torch.bfloat16)

def load_model():
    
    print(f"Loading chat model on {DEVICE}...")
    load_device = 'cpu' if DEVICE == 'mps' else DEVICE
    checkpoint = torch.load("data/best_checkpoint_chat.pt", map_location=load_device)
    
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
    return Tokenizer("data/tokenizer.model")  # Using chat-specific tokenizer

model = load_model()
tokenizer = get_tokenizer() 