from google import genai
from google.genai import types
import os

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
            
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            tools=tools,
            temperature=0.0
        )
    )
    return response

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
