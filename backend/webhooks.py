import os
import hmac
import hashlib
import random
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional, List
from github import Github, GithubException

from ingestion import fetch_all_files, fetch_specific_files, chunk_files
from embeddings import get_namespace, store_chunks, delete_file_chunks, has_namespace

webhook_router = APIRouter()

# GitHub config — secret is mandatory; refuse to start without it
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
if not GITHUB_WEBHOOK_SECRET:
    raise RuntimeError(
        "GITHUB_WEBHOOK_SECRET is not set. "
        "Refusing to start — unauthenticated webhooks would allow anyone to trigger reviews."
    )
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
gh_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

INGEST_API_KEY = os.getenv("INGEST_API_KEY")
if not INGEST_API_KEY:
    print("WARNING: INGEST_API_KEY is not set. /ingest endpoint will reject all requests.")

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

def verify_api_key(authorization: str) -> bool:
    """Verifies the Bearer token matches the configured INGEST_API_KEY."""
    if not INGEST_API_KEY or not authorization:
        return False
    if not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:], INGEST_API_KEY)

class IngestRequest(BaseModel):
    repo_url: str
    pr_diff: Optional[List[str]] = None

@webhook_router.post("/ingest")
async def ingest_endpoint(
    request: IngestRequest,
    authorization: str = Header(None),
):
    """Manual endpoint for full or incremental repo ingestion. Requires API key."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

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


async def _process_pr_review(repo_full_name: str, pr_number: int, files_changed: list):
    """Background task: ingestion + multi-agent review + GitHub comment."""
    from agent import run_agents_in_parallel

    repo_url = f"https://github.com/{repo_full_name}"
    namespace = get_namespace(repo_url)

    # --- AUTO INCREMENTAL INGESTION ---
    print(f"[BG] Checking if {namespace} exists in Pinecone...")
    if not has_namespace(namespace):
        print(f"[BG] First time seeing {repo_url}! Running FULL repo ingestion...")
        all_files = fetch_all_files(repo_url)
        chunks = chunk_files(all_files)
        if chunks:
            store_chunks(chunks, namespace)
        print(f"[BG] Full ingestion complete. Stored {len(chunks)} chunks for entire repo.")
    else:
        changed_file_names = [f["filename"] for f in files_changed if f["status"] != "removed"]
        removed_file_names = [f["filename"] for f in files_changed if f["status"] == "removed"]

        print(f"[BG] Running auto-incremental ingestion for {repo_url}...")
        for file_path in changed_file_names + removed_file_names:
            delete_file_chunks(file_path, namespace)

        updated_files = fetch_specific_files(repo_url, changed_file_names)
        chunks = chunk_files(updated_files)
        if chunks:
            store_chunks(chunks, namespace)

        print(f"[BG] Auto-ingestion complete. Stored {len(chunks)} chunks for {len(changed_file_names)} files.")

    # --- MULTI-AGENT REVIEW ---
    pr_diff_overview = ""
    for file_data in files_changed:
        pr_diff_overview += f"\n--- {file_data['filename']} ---\n{file_data.get('patch', 'No diff extracted')}\n"

    print(f"[BG] Triggering Core Orchestrator reasoning cycle for {repo_url}...")
    final_review = await run_agents_in_parallel(
        repo_url=repo_url,
        pr_diff=pr_diff_overview
    )

    print("\n\n====== FINAL AGENT PR REVIEW ======\n")
    print(final_review)
    print("\n===================================\n")

    # --- POST GITHUB COMMENT (with retry) ---
    MAX_RETRIES = 3
    BASE_DELAY = 2  # seconds

    if not gh_client:
        print("[BG] Warning: gh_client not initialized. Cannot post GitHub comment.")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[BG] Posting review to PR #{pr_number} (attempt {attempt}/{MAX_RETRIES})...")
            repo = gh_client.get_repo(repo_full_name)
            pull = repo.get_pull(pr_number)
            pull.create_issue_comment(final_review)
            print("[BG] Successfully posted PR comment!")
            break
        except GithubException as e:
            status = e.status
            is_transient = status >= 500 or status == 429
            if is_transient and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"[BG] GitHub API error {status} on attempt {attempt}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"[BG] GitHub API error {status} on attempt {attempt}. {'Non-retryable.' if not is_transient else 'Max retries exhausted.'}")
                print(f"[BG] Failed to post review: {e.data}")
                break
        except Exception as e:
            # Connection resets, timeouts, DNS failures, etc.
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"[BG] Connection error on attempt {attempt}: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"[BG] Connection error on attempt {attempt}: {e}. Max retries exhausted.")
                break


@webhook_router.post("/webhook", status_code=202)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None),
):
    payload_body = await request.body()
    
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
                background_tasks.add_task(_process_pr_review, repo_full_name, pr_number, files_changed)

            return {"status": "accepted", "message": "PR review queued", "metadata": pr_metadata}
            
    return {"status": "ignored", "message": f"Ignored event: {event}"}
