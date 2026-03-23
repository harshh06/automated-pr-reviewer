import os
import hmac
import hashlib
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional, List
from github import Github

from ingestion import fetch_all_files, fetch_specific_files, chunk_files
from embeddings import get_namespace, store_chunks, delete_file_chunks, has_namespace

webhook_router = APIRouter()

# GitHub config
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
gh_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """Verifies the HMAC-SHA256 signature from GitHub."""
    if not signature_header or not GITHUB_WEBHOOK_SECRET:
        return False
    hash_object = hmac.new(
        GITHUB_WEBHOOK_SECRET.encode('utf-8'), 
        msg=payload_body, 
        digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    return hmac.compare_digest(expected_signature, signature_header)

class IngestRequest(BaseModel):
    repo_url: str
    pr_diff: Optional[List[str]] = None

@webhook_router.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """Manual endpoint for full or incremental repo ingestion."""
    repo_url = request.repo_url
    namespace = get_namespace(repo_url)
    
    if not request.pr_diff:
        print(f"Starting FULL ingestion for {repo_url}...")
        files = fetch_all_files(repo_url)
        mode = "full"
    else:
        print(f"Starting INCREMENTAL ingestion for {repo_url}...")
        changed_files = request.pr_diff
        
        for file_path in changed_files:
            delete_file_chunks(file_path, namespace)
            
        files = fetch_specific_files(repo_url, changed_files)
        mode = "incremental"
        
    chunks = chunk_files(files)
    if chunks:
        store_chunks(chunks, namespace)
        print(f"Ingestion complete: {len(chunks)} chunks stored.")
        
    return {"status": "done", "chunks_stored": len(chunks), "mode": mode}

@webhook_router.post("/webhook")
async def github_webhook(request: Request, x_hub_signature_256: str = Header(None)):
    payload_body = await request.body()
    
    if GITHUB_WEBHOOK_SECRET:
        if not verify_signature(payload_body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    event = request.headers.get("X-GitHub-Event")
    
    if event == "pull_request":
        action = payload.get("action")
        if action in ["opened", "synchronize", "reopened"]:
            repo_full_name = payload["repository"]["full_name"]
            pr_number = payload["pull_request"]["number"]
            author = payload["pull_request"]["user"]["login"]
            branch = payload["pull_request"]["head"]["ref"]
            
            files_changed = []
            if gh_client:
                repo = gh_client.get_repo(repo_full_name)
                pr = repo.get_pull(pr_number)
                
                for f in pr.get_files():
                    files_changed.append({
                        "filename": f.filename,
                        "status": f.status,
                        "additions": f.additions,
                        "deletions": f.deletions,
                        "patch": getattr(f, "patch", "")
                    })

            pr_metadata = {
                "repo": repo_full_name,
                "pr_number": pr_number,
                "author": author,
                "branch": branch,
                "files_changed_count": len(files_changed),
                "files": [f["filename"] for f in files_changed]
            }
            
            print(f"Processed PR #{pr_number} from {repo_full_name} by {author}.")
            if files_changed:
                print(f"-> Extracted {len(files_changed)} files changed: {pr_metadata['files']}")
                
                # --- AUTO INCREMENTAL INGESTION ---
                repo_url = f"https://github.com/{repo_full_name}"
                namespace = get_namespace(repo_url)
                
                print(f"Checking if {namespace} exists in Pinecone...")
                if not has_namespace(namespace):
                    print(f"First time seeing {repo_url}! Running FULL repo ingestion...")
                    all_files = fetch_all_files(repo_url)
                    chunks = chunk_files(all_files)
                    if chunks:
                        store_chunks(chunks, namespace)
                    print(f"Full ingestion complete. Stored {len(chunks)} chunks for entire repo.")
                else:
                    changed_file_names = [f["filename"] for f in files_changed if f["status"] != "removed"]
                    removed_file_names = [f["filename"] for f in files_changed if f["status"] == "removed"]
                    
                    print(f"Running auto-incremental ingestion for {repo_url}...")
                    for file_path in changed_file_names + removed_file_names:
                        delete_file_chunks(file_path, namespace)
                        
                    updated_files = fetch_specific_files(repo_url, changed_file_names)
                    chunks = chunk_files(updated_files)
                    if chunks:
                        store_chunks(chunks, namespace)
                        
                    print(f"Auto-ingestion complete. Stored {len(chunks)} chunks for {len(changed_file_names)} files.")

                from agent import run_single_agent
                
                pr_diff_overview = ""
                for file_data in files_changed:
                    pr_diff_overview += f"\n--- {file_data['filename']} ---\n{file_data.get('patch', 'No diff extracted')}\n"
                    
                print(f"Triggering Core Agent reasoning cycle for {repo_url}...")
                final_review = run_single_agent(
                    repo_url=repo_url,
                    pr_diff=pr_diff_overview,
                    question="Please evaluate ONLY the specific code lines modified in the PR diff for security, performance, and code quality issues. Do NOT review the entire repository."
                )
                
                print("\n\n====== FINAL AGENT PR REVIEW ======\n")
                print(final_review)
                print("\n===================================\n")

            return {"status": "success", "message": "PR processed", "metadata": pr_metadata}
            
    return {"status": "ignored", "message": f"Ignored event: {event}"}
