import json
import torch
import asyncio
import uvicorn
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

#####################
###     SETUP     ###
#####################
from model_singleton import model, tokenizer, DEVICE, CTX, DEMO_MODE

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

request_queue = asyncio.Queue()
MAX_NEW_TOKENS = 300
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

#####################
###      FUNS     ###
#####################
def save_conversation(user_id: str, user_text: str, response: str, is_new_chat: bool = False):
    """Save conversation to a JSON file, appending to existing user file if it exists."""
    file_path = CONVERSATIONS_DIR / f"{user_id}.json"
    
    # Load existing conversations or create new list
    if file_path.exists():
        with open(file_path, "r") as f:
            conversations = json.load(f)
    else:
        conversations = []
    
    # Add new conversation
    conversation = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "user_message": user_text,
        "bot_response": response
    }
    conversations.append(conversation)
    
    # Save updated conversations
    with open(file_path, "w") as f:
        json.dump(conversations, f, indent=2)

async def get_demo_response(user_text: str) -> str:
    """Generate a demo response by echoing the input with some modifications."""
    # Split into words and reverse them
    words = user_text.split()
    reversed_words = words[::-1]
    response = " ".join(reversed_words)
    return response

async def model_worker():
    # willingly be.
    while True:
        # Each item is (user_text, response_queue, should_save, user_id, is_new_chat)
        user_text, response_queue, should_save, user_id, is_new_chat = await request_queue.get()
        try:

            user_text_reversed = " ".join(user_text.split()[::-1])
            print(user_text_reversed)
            reversed_input = f"<|response|> {user_text_reversed} <|instruction|> " #GENERATED TEXT <|endoftext|> "
            print(reversed_input)
            input_ids = tokenizer.encode(reversed_input, bos=True, eos=False)
            x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            generated_tokens = []
            full_response = ""
            max_new_tokens = MAX_NEW_TOKENS

            # Generate tokens one by one up to the maximum limit
            for _ in range(max_new_tokens):
                # Disable gradient computation for inference
                with torch.no_grad():
                    # Use the model context for efficient memory management
                    with CTX:
                        # Get model predictions (logits) for the current input
                        logits, _ = model(x)
                        # Apply temperature scaling (0.8) to control randomness
                        logits = logits[:, -1, :] / 0.8 
                        # Convert logits to probabilities using softmax
                        probs = torch.softmax(logits, dim=-1)
                        # Sample the next token from the probability distribution
                        next_token = torch.multinomial(probs, num_samples=1)
                # Stop generation if end-of-sequence token is encountered
                if next_token.item() == tokenizer.eos_id:
                    break
                # Decode the token back to text
                token_text = tokenizer.decode([next_token.item()])
                # Only process non-empty tokens
                if token_text.strip():
                    # Add token to the list of generated tokens
                    generated_tokens.append(next_token.item())
                    # Decode all generated tokens so far
                    current_text = tokenizer.decode(generated_tokens)
                    # Check if we've hit the instruction marker and stop if so
                    if "<|instruction|>" in current_text:
                        break
                    # Skip tokens that are part of the response marker
                    if "<|response|>" not in current_text:
                        # Send token to response queue for streaming
                        await response_queue.put(token_text)
                        # await asyncio.sleep(0.1)
                        # Accumulate the full response text
                        full_response += token_text
                # Append the new token to the input sequence for next iteration
                x = torch.cat([x, next_token], dim=1)
            
            if should_save: save_conversation(user_id, user_text, full_response, is_new_chat)    
            await response_queue.put(None)  # Signal end of stream
        except Exception as e:
            print(f"Error in model worker: {e}")
            await response_queue.put(None)
        finally:
            request_queue.task_done()

#####################
###     ROUTES    ###
#####################

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "demo_mode": DEMO_MODE
    })

@app.get("/user-id")
async def get_user_id():
    """Generate a new user ID."""
    user_id = str(uuid.uuid4())
    return JSONResponse({"user_id": user_id})

@app.post("/chat")
async def chat_stream(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    if not user_id:
        return JSONResponse({"error": "user_id is required"}, status_code=400)
        
    user_text = data["message"].strip()
    should_save = data.get("save_conversation", False)  # Default to False
    is_new_chat = data.get("is_new_chat", False)  # Default to False

    response_queue = asyncio.Queue()
    await request_queue.put((user_text, response_queue, should_save, user_id, is_new_chat))
    
    async def token_stream():
        while True:
            token = await response_queue.get()
            if token is None:
                break
            yield f"data: {token}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.get("/")
async def home(request: Request):
    return FileResponse("static/interface.html")

@app.on_event("startup")
async def startup_event():
    print("BACKCHAT SERVER STARTED")
    if DEMO_MODE: print("Running in DEMO mode")
    asyncio.create_task(model_worker())

@app.on_event("shutdown")
async def shutdown_event():
    print("BACKCHAT SERVER SHUTDOWN")

#####################
###     BEGIN     ###
#####################
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
