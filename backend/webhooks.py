import os
import hmac
import hashlib
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional, List
from github import Github, Auth

from redis_client import redis_client
from quota import check_user_quota, MAX_PRS_PER_USER
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
# GitHub App config
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
GITHUB_APP_PRIVATE_KEY = os.getenv("GITHUB_APP_PRIVATE_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_github_client(installation_id: Optional[int] = None) -> Github:
    """
    Returns a dynamically authenticated PyGithub client.
    If App ID, Private Key, and installation_id are present, acts as GitHub App.
    Otherwise, falls back to the old GITHUB_TOKEN logic.
    """
    if GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY and installation_id:
        try:
            # Handle both filepath or raw string for private key
            private_key = GITHUB_APP_PRIVATE_KEY
            if os.path.exists(GITHUB_APP_PRIVATE_KEY):
                with open(GITHUB_APP_PRIVATE_KEY, 'r') as f:
                    private_key = f.read()

            app_auth = Auth.AppAuth(app_id=GITHUB_APP_ID, private_key=private_key)
            # PyGithub magically fetches the installation access token
            installation_auth = app_auth.get_installation_auth(installation_id)
            return Github(auth=installation_auth)
        except Exception as e:
            print(f"Failed to authenticate as GitHub App: {e}")
            print("Falling back to GITHUB_TOKEN...")
            
    # Fallback to Personal Access Token
    return Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

INGEST_API_KEY = os.getenv("INGEST_API_KEY")
if not INGEST_API_KEY:
    print("WARNING: INGEST_API_KEY is not set. /ingest endpoint will reject all requests.")

MAX_PR_FILES = 10
MAX_PR_LINES = 300

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



@webhook_router.post("/webhook", status_code=202)
async def github_webhook(
    request: Request,
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
            sha = payload["pull_request"]["head"]["sha"]
            installation_id = payload.get("installation", {}).get("id")

            # --- 1. IDEMPOTENCY CHECK ---
            # Prevent duplicate jobs when GitHub re-pings the same webhook delivery.
            # Key is scoped to repo + PR number + commit SHA so re-opens and new
            # pushes to the same PR still trigger a fresh review.
            idempotency_key = f"pr_review:{repo_full_name}:{pr_number}:{sha}"
            already_queued = not redis_client.set(idempotency_key, "1", ex=3600, nx=True)
            if already_queued:
                print(f"[Webhook] Duplicate delivery for PR#{pr_number} sha={sha[:7]} — skipping.")
                return {"status": "already_queued", "message": "Review already in progress for this commit"}

            files_changed = []
            total_lines = 0

            gh_client = get_github_client(installation_id)
            if gh_client:
                repo = gh_client.get_repo(repo_full_name)
                pr = repo.get_pull(pr_number)

                # --- 2. USER QUOTA CHECK ---
                if not check_user_quota(author):
                    msg = (
                        f"👋 Hi @{author}! **PR Sentinel** is currently a free demo project. "
                        f"To prevent API abuse, guests are limited to {MAX_PRS_PER_USER} automated reviews. "
                        f"You've reached the limit!"
                    )
                    pr.create_issue_comment(msg)
                    print(f"Skipped PR #{pr_number}: User @{author} exceeded quota.")
                    # Release the idempotency key — quota block is not a "review in progress"
                    redis_client.delete(idempotency_key)
                    return {"status": "ignored", "message": "User quota exceeded"}

                for f in pr.get_files():
                    files_changed.append({
                        "filename": f.filename,
                        "status": f.status,
                        "additions": f.additions,
                        "deletions": f.deletions,
                        "patch": getattr(f, "patch", ""),
                    })
                    total_lines += f.additions + f.deletions

                # --- 3. PR SIZE LIMIT CHECK ---
                if author != "harshh06" and (
                    len(files_changed) > MAX_PR_FILES or total_lines > MAX_PR_LINES
                ):
                    msg = (
                        f"👋 Hi @{author}! **PR Sentinel** is running as a free public demo. "
                        f"To protect API limits, I can only review small PRs "
                        f"(Under {MAX_PR_FILES} files and {MAX_PR_LINES} lines).\n\n"
                        f"This PR has **{len(files_changed)} files** and **{total_lines} lines** changed. "
                        f"Please submit a smaller PR!"
                    )
                    pr.create_issue_comment(msg)
                    print(f"Skipped PR #{pr_number}: Too large ({len(files_changed)} files, {total_lines} lines).")
                    redis_client.delete(idempotency_key)
                    return {"status": "ignored", "message": "PR too large"}

            pr_metadata = {
                "repo": repo_full_name,
                "pr_number": pr_number,
                "author": author,
                "branch": branch,
                "sha": sha[:7],
                "files_changed_count": len(files_changed),
                "files": [f["filename"] for f in files_changed],
            }

            # --- 4. ENQUEUE TO CELERY (durable, survives restarts) ---
            if files_changed:
                from tasks import ingest_task
                ingest_task.delay(repo_full_name, pr_number, files_changed, sha, installation_id)
                print(
                    f"[Webhook] Enqueued ingest_task for PR#{pr_number} "
                    f"from {repo_full_name} by {author}. "
                    f"Files: {pr_metadata['files']}"
                )

            return {"status": "accepted", "message": "PR review queued", "metadata": pr_metadata}

    return {"status": "ignored", "message": f"Ignored event: {event}"}
