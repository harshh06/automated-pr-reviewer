import os
from google import genai

# Try to load the client if the key is available
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print("Warning: Could not initialize genai.Client.", e)

DEFAULT_MODEL = 'gemini-2.5-flash'

def call_llm(messages, tools=None):
    """
    High-level entry point to call the LLM.
    For Gemini with automatic tool calls, this handles the loop behind the scenes.
    """
    if not client:
        raise ValueError("LLM Client not initialized. Check GEMINI_API_KEY.")
        
    config = {}
    if tools:
        config['tools'] = tools
        
    return client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=messages,
        config=config
    )

def extract_text(response) -> str:
    """Extract text from the response."""
    if hasattr(response, "text"):
        return response.text
    return str(response)

# Stubs for manual tool calling loop (useful when switching to Claude)
def parse_tool_call(response):
    """Extract a tool call from the response if present."""
    pass

def format_tool_result(tool_use_id, result):
    """Format the tool execution result to be sent back to the LLM."""
    pass
