import os
from github import Github

# Filter extensions for code ingestion
ALLOWED_EXTENSIONS = {".py", ".ts", ".js", ".tsx", ".java", ".go", ".rb", ".cpp", ".md"}
IGNORED_DIRS = {"node_modules", ".git", "build", "dist", "__pycache__"}

# Approximating 500-1000 tokens using character length
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
gh_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

def is_allowed(file_path: str):
    # Check directory
    if any(ignored in file_path.split('/') for ignored in IGNORED_DIRS):
        return False
    # Check extension
    _, ext = os.path.splitext(file_path)
    if ext not in ALLOWED_EXTENSIONS:
        return False
    return True

def fetch_all_files(repo_url: str):
    """Recursively fetch repository files via PyGithub for full ingestion."""
    if not gh_client:
        print("WARNING: GITHUB_TOKEN not set. Cannot fetch all files.")
        return []
    
    repo_name = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
    repo = gh_client.get_repo(repo_name)
    
    files = []
    try:
        default_branch = repo.default_branch
        # Using git_tree for recursive fetch is way more API-efficient than get_contents("") recursively
        tree = repo.get_git_tree(default_branch, recursive=True)
        for element in tree.tree:
            if element.type == "blob" and is_allowed(element.path):
                file_content = repo.get_contents(element.path)
                try:
                    text = file_content.decoded_content.decode("utf-8")
                    files.append({"file_path": element.path, "content": text})
                    print(f"Fetched {element.path}")
                except Exception as e:
                    print(f"Skipping {element.path} due to decode error: {e}")
    except Exception as e:
        print(f"Error fetching full repo tree: {e}")
        
    return files

def fetch_specific_files(repo_url: str, changed_files: list):
    """Fetch only the specific changed files for incremental indexing."""
    if not gh_client:
        return []
        
    repo_name = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
    repo = gh_client.get_repo(repo_name)
    
    files = []
    for file_path in changed_files:
        if is_allowed(file_path):
            try:
                file_content = repo.get_contents(file_path)
                text = file_content.decoded_content.decode("utf-8")
                files.append({"file_path": file_path, "content": text})
            except Exception as e:
                print(f"Skipped {file_path}: {e}")
    return files

def chunk_files(files: list):
    """Chunks documents into overlapping pieces."""
    chunks = []
    for f in files:
        file_path = f["file_path"]
        content = f["content"]
        
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + CHUNK_SIZE, content_length)
            chunk_text = content[start:end]
            
            # Approximate line numbers based on character offsets
            start_line = content.count('\n', 0, start) + 1
            end_line = content.count('\n', 0, end) + 1
            
            chunks.append({
                "file_path": file_path,
                "content": f"File: {file_path}\n" + chunk_text,
                "start_line": start_line,
                "end_line": end_line
            })
            
            if start + CHUNK_SIZE >= content_length:
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
            
    return chunks
