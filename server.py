from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from pathlib import Path
import asyncio
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and tokenizer
def load_model():
    checkpoint = torch.load("out/best_checkpoint.pt", map_location="cuda")
    model_config = GPTConfig(**checkpoint["model_args"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to("cuda")
    model.eval()
    return model

def get_tokenizer():
    return Tokenizer("data/tok4096.model")

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
        x = torch.tensor(input_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        
        # Generate until EOS or max length (safety limit)
        max_new_tokens = 200  # Increased safety limit
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
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
                    await asyncio.sleep(0.05)  # Add slight delay for visual effect
                
                x = torch.cat([x, next_token], dim=1)
    
    return StreamingResponse(generate(), media_type="text/event-stream") 