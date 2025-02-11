from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from pathlib import Path
import asyncio
from model import GPT
from config_fine10 import GPTConfig
from tokenizer import Tokenizer
from contextlib import nullcontext

app = FastAPI()

# Mount static files - using the same static and template files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Determine device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Setup context manager for the device
if DEVICE == "cuda":
    CTX = torch.autocast(DEVICE, dtype=torch.bfloat16)
elif DEVICE == "mps":
    CTX = nullcontext()  # No autocast for MPS
else:
    CTX = torch.autocast("cpu", dtype=torch.bfloat16)

# Load model and tokenizer
def load_model():
    print(f"Loading fine10 model on {DEVICE}...")
    load_device = 'cpu' if DEVICE == 'mps' else DEVICE
    checkpoint = torch.load("out/checkpoints_fine10/best_checkpoint_fine10.pt", map_location=load_device)
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
    return Tokenizer("data/tok8000_fine10.model")

model = load_model()
tokenizer = get_tokenizer()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

@app.post("/chat")
async def chat_stream(request: Request):
    data = await request.json()
    user_text = data["message"].strip()
    
    async def generate():
        # Reverse the input text for our backwards model
        words = user_text.split()
        reversed_input = " ".join(words[::-1])
        
        # Encode the reversed input
        input_ids = tokenizer.encode(reversed_input, bos=True, eos=False)
        x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        # Generate until EOS or max length (safety limit)
        max_new_tokens = 200  # Same safety limit as main server
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                with CTX:
                    logits, _ = model(x)
                    logits = logits[:, -1, :] / 0.7  # temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Break if we hit the EOS token
                if next_token.item() == tokenizer.eos_id:
                    break
                    
                # Decode the single token
                token_text = tokenizer.decode([next_token.item()])
                if token_text.strip():  # Only yield non-empty tokens
                    yield f"data: {token_text}\n\n"
                    await asyncio.sleep(0.01)  # Small delay between tokens
                x = torch.cat([x, next_token], dim=1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# Run with: uvicorn server_fine:app --host 0.0.0.0 --port 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)