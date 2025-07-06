import torch
from contextlib import nullcontext

from model import GPT
from config import BackStoryConfig
from tokenizer import BackStoryTokenizer, BACK_STORY_TOKENIZER_MODEL


if torch.cuda.is_available(): DEVICE = "cuda"
# elif torch.backends.mps.is_available():  DEVICE = "mps"
else: DEVICE = "cpu"

if DEVICE == "cuda":  CTX = torch.autocast(DEVICE, dtype=torch.bfloat16) 
elif DEVICE == "mps": CTX = nullcontext()  # No autocast for MPS
else: CTX = torch.autocast("cpu", dtype=torch.bfloat16)

# Load model and tokenizer
def load_model():
    print(f"Loading model on {DEVICE}...")
    checkpoint = torch.load("data/backstory_best_checkpoint.pt", map_location=DEVICE)
    model_config = BackStoryConfig(**checkpoint["model_args"])
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
    return BackStoryTokenizer(BACK_STORY_TOKENIZER_MODEL)

model = load_model()
tokenizer = get_tokenizer()