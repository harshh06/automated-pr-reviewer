from embeddings import search
from ingestion import fetch_specific_files, gh_client, IGNORED_DIRS, is_allowed
import os

# Store the global tracking context required by the tools
_CURRENT_NAMESPACE = ""
_CURRENT_REPO_URL = ""

def set_tool_context(namespace: str, repo_url: str):
    global _CURRENT_NAMESPACE, _CURRENT_REPO_URL
    _CURRENT_NAMESPACE = namespace
    _CURRENT_REPO_URL = repo_url

def search_codebase(query: str) -> str:
    """Semantic search over the repository's Pinecone embeddings. Returns 5 closest code chunks matching the query string semantics. Use this first to find where things might be."""
    if not _CURRENT_NAMESPACE:
        return "Error: Repository namespace not set."
    results = search(query, _CURRENT_NAMESPACE, top_k=5)
    if not results:
        return "No semantic matches found."
    
    output = []
    for i, r in enumerate(results):
        snippet = r['content']
        output.append(f"[Result {i+1}] File: {r['file_path']} (Line {r['start_line']})\n{snippet}")
    return "\n\n---\n\n".join(output)

def read_file(file_path: str) -> str:
    """Fetches the full text content of a specific file in the repository. Provide the full relative string filepath."""
    if not _CURRENT_REPO_URL:
        return "Error: Repository URL not set."
    
    files = fetch_specific_files(_CURRENT_REPO_URL, [file_path])
    if not files:
        return f"File '{file_path}' not found or could not be read."
    return files[0]["content"]

def list_files(directory: str) -> str:
    """Lists all files cleanly inside a specific repository directory. Pass empty string '' for the root directory."""
    if not gh_client:
        return "GitHub client not initialized."
    if not _CURRENT_REPO_URL:
        return "Repository URL not set."
        
    repo_name = _CURRENT_REPO_URL.replace("https://github.com/", "")
    repo = gh_client.get_repo(repo_name)
    
    try:
        contents = repo.get_contents(directory) if directory else repo.get_contents("")
        
        if not isinstance(contents, list):
            contents = [contents]
            
        file_list = []
        for element in contents:
            if element.type == "dir" and element.name not in IGNORED_DIRS:
                file_list.append(f"[DIR]  {element.path}")
            elif element.type == "blob" and is_allowed(element.path):
                file_list.append(f"[FILE] {element.path}")
                
        return "\n".join(file_list) if file_list else f"Directory '{directory}' is empty."
    except Exception as e:
        return f"Error listing directory '{directory}': {e}"

def grep_code(pattern: str) -> str:
    """Exact string search (substring) across the entire indexed repository code. Enter a short string to search for occurrences natively across GitHub."""
    if not _CURRENT_REPO_URL:
        return "Error: Repository URL not set."
        
    try:
        repo_name = _CURRENT_REPO_URL.replace("https://github.com/", "")
        
        # We leverage GitHub native Search API
        results = gh_client.search_code(query=f"{pattern} repo:{repo_name}")
        
        output = []
        count = 0
        for r in results:
            if count >= 10:
                break
            output.append(f"- {r.path}")
            count += 1
            
        if not output:
            return f"No occurrences found for exactly: {pattern}"
        return f"Found exact occurrences in {count} files:\n" + "\n".join(output) + "\n(Note: Use read_file tool to view the contents of these files.)"
        
    except Exception as e:
        return f"Error executing grep: {e}"

agent_tools = [search_codebase, read_file, list_files, grep_code]
