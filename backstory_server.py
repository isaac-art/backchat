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
from backstory import model, tokenizer, DEVICE, CTX

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

async def model_worker():
    while True:
        # Each item is (user_text, response_queue, should_save, user_id, is_new_chat)
        user_text, response_queue, should_save, user_id, is_new_chat = await request_queue.get()
        try:
            # Reverse the input text for our backwards model
            words = user_text.split()
            reversed_input = " ".join(words[::-1])
            input_ids = tokenizer.encode(reversed_input, bos=True, eos=False)
            x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            generated_tokens = []
            full_response = ""
            max_new_tokens = MAX_NEW_TOKENS

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    with CTX:
                        logits, _ = model(x)
                        logits = logits[:, -1, :] / 0.7  # temperature - higher is more random
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == tokenizer.eos_id:
                    break
                token_text = tokenizer.decode([next_token.item()])
                if token_text.strip():
                    generated_tokens.append(next_token.item())
                    await response_queue.put(token_text)
                    await asyncio.sleep(0.01)
                    full_response += token_text
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
        "device": DEVICE
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
            await asyncio.sleep(0.01)
    return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.get("/")
async def home(request: Request):
    return FileResponse("static/story_interface.html")

@app.on_event("startup")
async def startup_event():
    print("BACKCHAT SERVER STARTED")
    asyncio.create_task(model_worker())

@app.on_event("shutdown")
async def shutdown_event():
    print("BACKCHAT SERVER SHUTDOWN")

#####################
###     BEGIN     ###
#####################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
