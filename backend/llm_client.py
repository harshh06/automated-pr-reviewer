from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv
from google.genai.errors import ClientError
import random

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=api_key) if api_key else None

def call_llm(messages, tools):
    """
    Standard interface to call the LLM regardless of provider.
    messages: list of standard dictionaries {"role": ..., "content": ..., etc}
    tools: list of raw python functions
    """
    if not client:
        raise Exception("No Gemini client configured.")
    
    gemini_contents = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
        elif role == "assistant":
            if msg.get("tool_calls"):
                parts = []
                for tc in msg["tool_calls"]:
                    parts.append(types.Part.from_function_call(
                        name=tc["name"], 
                        args=tc["args"]
                    ))
                gemini_contents.append(types.Content(role="model", parts=parts))
            else:
                gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg.get("content", ""))]))
        elif role == "tool":
            gemini_contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=msg["tool_name"],
                    response={"result": msg["content"]}
                )]
            ))
            
    # Try up to 5 times dynamically for strict Free Tier limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    tools=tools if tools else None
                )
            )
            return response
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    jitter = random.randint(10, 50)
                    print(f"⚠️ Google API Limit hit ({attempt + 1}/{max_retries}). Coasting for {60 + jitter}s to stagger threads...")
                    time.sleep(60 + jitter)
                else:
                    print("❌ Final Attempt failed. Google rate limits persist.")
                    raise e
            else:
                # If it's a completely unrelated error (e.g. 500 server crash), throw immediately.
                raise e

def parse_tool_call(response):
    """Extracts requested tool calls from the provider's response format."""
    if response and getattr(response, "function_calls", None):
        calls = []
        for fc in response.function_calls:
            calls.append({
                "id": fc.name, # Gemini function calls uniquely map back by name in the response
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {}
            })
        return calls
    return None

def format_tool_result(tool_name, result_str):
    """Formats the executed tool return value into our generic message format."""
    return {
        "role": "tool",
        "tool_name": tool_name,
        "content": str(result_str)
    }
