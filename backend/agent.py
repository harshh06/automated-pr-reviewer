from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import asyncio
import copy

from llm_client import call_llm, parse_tool_call, format_tool_result
from tools import agent_tools, set_tool_context
from embeddings import get_namespace

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    domain: str

def call_llm_node(state: AgentState):
    """The node that actually queries the LLM with the available messaging history and tools."""
    messages = state["messages"]
    domain = state.get("domain", "general")
    
    # We pass the shared array of tools
    response = call_llm(messages, tools=agent_tools)
    
    tool_calls = parse_tool_call(response)
    if tool_calls:
        # LLM requested an explicit function execution
        return {"messages": [{"role": "assistant", "tool_calls": tool_calls}]}
    else:
        # LLM generated final human string answer
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
            
            func_map = {f.__name__: f for f in agent_tools}
            func = func_map.get(tool_name)
            
            if func:
                try:
                    print(f"[{state.get('domain', 'Agent').upper()}] Executing tool '{tool_name}'...")
                    result = func(**tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            else:
                result = f"Unknown tool: {tool_name}"
                
            tool_results.append(format_tool_result(tool_name, result))
            
    return {"messages": tool_results}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "call_tool"
    return END

def create_specialized_agent():
    """Factory function cloning independent isolated LangGraph instances dynamically."""
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm_node)
    graph.add_node("tool", call_tool_node)
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tool", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


async def run_agents_in_parallel(repo_url: str, pr_diff: str) -> str:
    """Orchestrator Node managing Fan-out and Aggregation behavior."""
    namespace = get_namespace(repo_url)
    
    # Establish the context so Python tools know what repo they are pulling from
    set_tool_context(namespace, repo_url)

    # Fresh executors per review — no shared state across concurrent PRs
    security_agent_executor = create_specialized_agent()
    performance_agent_executor = create_specialized_agent()
    quality_agent_executor = create_specialized_agent()
    
    print(f"\n--- Booting MULTI-AGENT ORCHESTRATOR for {repo_url} ---")
    
    sec_instructions = (
        "You are a strict Security Expert reviewing a Pull Request. Focus SOLELY on: SQL injections, hardcoded keys, unvalidated input, insecure dependencies, auth flaws.\n"
        f"You have access to `{repo_url}` via tools. CRITICAL: Never guess context. Use `read_file` or `search_codebase` specifically around the modified diff lines before speaking.\n"
        f"PR Diff:\n```diff\n{pr_diff}\n```\n\nAnalyze this explicitly for solely Security vulnerabilities."
    )
    perf_instructions = (
        "You are a strict Performance Expert reviewing a Pull Request. Focus SOLELY on: N+1 queries, unindexed searches, inefficient recursive loops, heavy memory allocations.\n"
        f"You have access to `{repo_url}` via tools. CRITICAL: Never guess context. Use `read_file` or `search_codebase` specifically around the modified diff lines before speaking.\n"
        f"PR Diff:\n```diff\n{pr_diff}\n```\n\nAnalyze this explicitly for solely Performance blockages."
    )
    qual_instructions = (
        "You are a strict Code Quality Expert reviewing a Pull Request. Focus SOLELY on: Duplicate logic, terrible naming conventions, missing tests, absent error handling schemas.\n"
        f"You have access to `{repo_url}` via tools. CRITICAL: Never guess context. Use `read_file` or `search_codebase` specifically around the modified diff lines before speaking.\n"
        f"PR Diff:\n```diff\n{pr_diff}\n```\n\nAnalyze this explicitly for solely Code Quality and readability improvements."
    )
    
    print("> Orchestrator: Dispatching sub-agents asynchronously...")
    # Fire all three isolated multi-turn graphs completely parallel asynchronously!
    sec_task = security_agent_executor.ainvoke({"messages": [{"role": "user", "content": sec_instructions}], "domain": "Security"})
    perf_task = performance_agent_executor.ainvoke({"messages": [{"role": "user", "content": perf_instructions}], "domain": "Performance"})
    qual_task = quality_agent_executor.ainvoke({"messages": [{"role": "user", "content": qual_instructions}], "domain": "Quality"})

    # Wait for all tools, searches, and reasoning blocks to complete globally.
    results = await asyncio.gather(sec_task, perf_task, qual_task)
    print("> Orchestrator: All sub-agents successfully completed! Aggregating...")

    sec_findings = results[0]["messages"][-1]["content"]
    perf_findings = results[1]["messages"][-1]["content"]
    qual_findings = results[2]["messages"][-1]["content"]

    # Final Orchestrator Sync (Formatting Node)
    aggregator_instruction = (
        "You are the Lead Master Orchestrator. Aggregate these three isolated unedited expert sub-agent reviews into ONE final structured GitHub Pull Request comment.\n"
        "Be extremely clean. Use markdown natively.\n"
        "Format headers exactly like this:\n## 🤖 Automated PR Review\n\n### 🔒 Security\n### ⚡ Performance\n### 📋 Code Quality\n\n"
        f"Raw Security Findings:\n{sec_findings}\n\n"
        f"Raw Performance Findings:\n{perf_findings}\n\n"
        f"Raw Code Quality Findings:\n{qual_findings}"
    )
    
    # Run the synchronous LLM call in a thread to keep the event loop responsive
    agg_response = await asyncio.to_thread(call_llm, [{"role": "user", "content": aggregator_instruction}], [])
    
    final_review = getattr(agg_response, "text", None)
    if final_review is None and getattr(agg_response, "candidates", None) and agg_response.candidates:
        parts = agg_response.candidates[0].content.parts
        final_review = "".join(getattr(p, "text", "") for p in parts)
            
    return final_review or "Error: Aggregation failed."
