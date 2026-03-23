from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

# Import our LLM adapter (this is the ONLY file we'll need to swap for Claude)
from llm_client import call_llm

# Import the webhooks router
from webhooks import webhook_router

app = FastAPI()

# Include the webhook router
app.include_router(webhook_router)

class ChatRequest(BaseModel):
    message: str

def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"The weather in {city} is Sunny and 75°F"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Use our adapter to handle the LLM interaction
        response = call_llm(
            messages=[{"role": "user", "content": request.message}],
            tools=[get_weather]
        )
        return {"response": getattr(response, "text", "No text generated.")}
    except Exception as e:
        print(f"Detailed Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)