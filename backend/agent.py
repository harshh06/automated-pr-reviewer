from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator

from llm_client import call_llm, parse_tool_call, format_tool_result
from tools import agent_tools, set_tool_context
from embeddings import get_namespace

# The state tracks the conversation between the user, LLM, and tools dynamically.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def call_llm_node(state: AgentState):
    """The node that actually queries the LLM with the available messaging history and tools."""
    messages = state["messages"]
    
    # Pass execution to our llm_client adapter securely decoupled
    response = call_llm(messages, tools=agent_tools)
    
    tool_calls = parse_tool_call(response)
    if tool_calls:
        # The LLM explicitly asked to use a specific function tool
        return {"messages": [{"role": "assistant", "tool_calls": tool_calls}]}
    else:
        # The LLM decided it has enough information to yield a final language answer!
        final_text = getattr(response, "text", None)
        if final_text is None and getattr(response, "candidates", None) and response.candidates:
            parts = response.candidates[0].content.parts
            final_text = "".join(getattr(p, "text", "") for p in parts)
            
        return {"messages": [{"role": "assistant", "content": final_text or ""}]}

def call_tool_node(state: AgentState):
    """The node that manually receives requests to run functions and acts as the runtime environment."""
    last_msg = state["messages"][-1]
    
    tool_results = []
    if "tool_calls" in last_msg:
        for tc in last_msg["tool_calls"]:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            
            # Map name to actual python function objects
            func_map = {f.__name__: f for f in agent_tools}
            func = func_map.get(tool_name)
            
            if func:
                try:
                    print(f"Executing tool '{tool_name}' with args {tool_args}...")
                    result = func(**tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            else:
                result = f"Unknown internal tool: {tool_name}"
                
            tool_results.append(format_tool_result(tool_name, result))
            
    return {"messages": tool_results}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "call_tool"
    return END

# Instantiate LangGraph Agent!
graph = StateGraph(AgentState)

# Attach the core cycle
graph.add_node("llm", call_llm_node)
graph.add_node("tool", call_tool_node)

# Add logic routers predicting loops
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tool", "llm")
graph.set_entry_point("llm")

# Compile
agent_executor = graph.compile()


def run_single_agent(repo_url: str, pr_diff: str, question: str = "Analyze this PR comprehensively.") -> str:
    """Entry point specifically for Milestone 4 standard single agent loop."""
    namespace = get_namespace(repo_url)
    set_tool_context(namespace, repo_url)
    
    instructions = (
        "You are an expert GitHub PR Reviewer. Your SOLE job is to strictly analyze the code changes present inside the `PR Diff Context`.\n"
        f"You have access to the full repository `{repo_url}` via tools. "
        "CRITICAL: Do NEVER output a comprehensive global security or performance audit of the whole application. ONLY review the specific lines modified, added, or deleted in the PR Diff. "
        "Use your codebase tools (`search_codebase`, `read_file`) ONLY to gather surrounding execution context about the specific lines that changed (e.g. 'Does this newly modified variable break a schema in another file?').\n\n"
        f"PR Diff Context:\n```diff\n{pr_diff}\n```\n\n"
        f"User Request:\n{question}"
    )
    
    state = {"messages": [{"role": "user", "content": instructions}]}
    
    print(f"\n--- Booting Agent for {repo_url} ---")
    
    for step in agent_executor.stream(state):
        if "llm" in step:
            last_msg = step["llm"]["messages"][-1]
            if "tool_calls" in last_msg:
                print(f"[LLM Agent] requested tools execution: {[tc['name'] for tc in last_msg['tool_calls']]}")
            else:
                print(f"[LLM Agent] successfully generated final analysis response.")
                return last_msg["content"]
        elif "tool" in step:
            print(f"[Environment] All tools executed successfully. Routing results back to LLM...")
            
    return ""
