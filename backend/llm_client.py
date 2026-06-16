from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from retry_utils import call_with_gemini_retry

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def call_llm(messages, tools):
    """
    Standard interface to call the LLM regardless of provider.

    Messages can be:
      - Plain dicts: {"role": "user"/"assistant"/"tool", "content": ..., ...}
      - Raw Gemini Content objects (preserved from a previous response to keep
        thought_signatures intact).

    Returns the raw Gemini response object.
    """
    if not client:
        raise Exception("No Gemini client configured.")

    gemini_contents = []
    for msg in messages:
        # If this is already a native Gemini Content object, pass it through
        # as-is.  This preserves thought_signatures on model function-call
        # parts, which Gemini 3 models require for multi-turn tool use.
        if isinstance(msg, types.Content):
            gemini_contents.append(msg)
            continue

        role = msg.get("role")

        # Skip LangGraph routing markers — the raw Content object (above)
        # already contains the function_call parts with thought_signatures.
        if role == "assistant" and "tool_calls" in msg:
            continue

        if role == "user":
            gemini_contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])])
            )
        elif role == "assistant":
            gemini_contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=msg.get("content", ""))])
            )
        elif role == "tool":
            gemini_contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=msg["tool_name"],
                    response={"result": msg["content"]}
                )]
            ))

    return call_with_gemini_retry(
        client.models.generate_content,
        model="gemini-2.5-flash-lite",
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            tools=tools if tools else None
        )
    )

def parse_tool_call(response):
    """Extracts requested tool calls from the provider's response format."""
    if response and getattr(response, "function_calls", None):
        calls = []
        for fc in response.function_calls:
            calls.append({
                "id": fc.name,
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {}
            })
        return calls
    return None


def get_raw_model_content(response):
    """
    Return the raw Content object from the model's response.

    This Content object contains the function_call parts WITH their
    thought_signatures, which must be echoed back on the next turn for
    Gemini 3+ models.
    """
    if response and getattr(response, "candidates", None) and response.candidates:
        return response.candidates[0].content
    return None


def format_tool_result(tool_name, result_str):
    """Formats the executed tool return value into our generic message format."""
    return {
        "role": "tool",
        "tool_name": tool_name,
        "content": str(result_str)
    }
