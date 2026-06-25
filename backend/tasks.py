"""
tasks.py
Two chained Celery tasks for durable PR review processing.

WHY TWO TASKS:
  ingest_task  — fetches repo files and stores in Pinecone.
                 Retries up to 3x on GitHub/Pinecone errors.
                 On success, enqueues review_task.

  review_task  — runs LangGraph multi-agent review and posts GitHub comment.
                 Retries up to 10x on Gemini errors (spans ~30 minutes via
                 exponential backoff).
                 Ingestion is NEVER re-run on retry — only agents retry.

This separation means a Gemini outage never triggers redundant Pinecone writes.
"""
import asyncio
import os
import sys
from typing import Optional
from celery import Task
from celery.utils.log import get_task_logger

# Celery forks worker subprocesses — they don't inherit the main process sys.path.
# Insert the backend directory explicitly so all project modules are importable.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)


from celery_app import celery_app
from redis_client import redis_client

logger = get_task_logger(__name__)

# ── Retry config ─────────────────────────────────────────────────────────────

INGEST_MAX_RETRIES = 3
REVIEW_MAX_RETRIES = 10
BASE_BACKOFF = 2  # seconds — doubled each retry (2, 4, 8, 16, 32...)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_github_client(installation_id):
    """Inline import to avoid circular deps at module load time."""
    from webhooks import get_github_client
    return get_github_client(installation_id)


def _build_pr_diff_overview(files_changed: list) -> str:
    """Reconstruct the unified diff string from stored file metadata."""
    overview = ""
    for file_data in files_changed:
        overview += f"\n--- {file_data['filename']} ---\n{file_data.get('patch', 'No diff extracted')}\n"
    return overview


# ── Task 1: Ingestion ─────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.ingest_task",
    max_retries=INGEST_MAX_RETRIES,
    default_retry_delay=BASE_BACKOFF,
)
def ingest_task(
    self: Task,
    repo_full_name: str,
    pr_number: int,
    files_changed: list,
    sha: str,
    installation_id: Optional[int],
):
    """
    Step 1: Ingest repo files into Pinecone.
    On success, chains to review_task automatically.
    """
    from ingestion import fetch_all_files, fetch_specific_files, chunk_files
    from embeddings import get_namespace, store_chunks, delete_file_chunks, has_namespace

    repo_url = f"https://github.com/{repo_full_name}"
    namespace = get_namespace(repo_url)
    gh_client = _get_github_client(installation_id)

    logger.info(f"[ingest_task] Starting for {repo_full_name} PR#{pr_number} sha={sha[:7]}")

    try:
        if not has_namespace(namespace):
            logger.info(f"[ingest_task] First time seeing {repo_url} — running FULL ingestion")
            all_files = fetch_all_files(repo_url, gh_client)
            chunks = chunk_files(all_files)
            if chunks:
                store_chunks(chunks, namespace)
            logger.info(f"[ingest_task] Full ingestion done: {len(chunks)} chunks stored")
        else:
            changed_file_names = [f["filename"] for f in files_changed if f["status"] != "removed"]
            removed_file_names = [f["filename"] for f in files_changed if f["status"] == "removed"]

            logger.info(f"[ingest_task] Incremental ingestion for {len(changed_file_names)} changed files")
            for file_path in changed_file_names + removed_file_names:
                delete_file_chunks(file_path, namespace)

            updated_files = fetch_specific_files(repo_url, changed_file_names, sha, gh_client)
            chunks = chunk_files(updated_files)
            if chunks:
                store_chunks(chunks, namespace)

            skipped = len(changed_file_names) - len(updated_files)
            logger.info(
                f"[ingest_task] Incremental done: {len(chunks)} chunks for "
                f"{len(updated_files)}/{len(changed_file_names)} files "
                f"({skipped} skipped)"
            )

    except Exception as exc:
        attempt = self.request.retries + 1
        wait = BASE_BACKOFF ** attempt
        logger.warning(
            f"[ingest_task] Attempt {attempt}/{INGEST_MAX_RETRIES} failed: {exc}. "
            f"Retrying in {wait}s..."
        )
        raise self.retry(exc=exc, countdown=wait)

    # ── Ingestion succeeded → chain to review ────────────────────────────────
    logger.info(f"[ingest_task] Done. Enqueuing review_task for PR#{pr_number}")
    review_task.delay(repo_full_name, pr_number, files_changed, sha, installation_id)


# ── Task 2: Review ────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.review_task",
    max_retries=REVIEW_MAX_RETRIES,
)
def review_task(
    self: Task,
    repo_full_name: str,
    pr_number: int,
    files_changed: list,
    sha: str,
    installation_id: Optional[int],
):
    """
    Step 2: Run multi-agent review and post GitHub comment.
    Retries up to 10x on Gemini failures — spans ~30 minutes via exponential backoff.
    Ingestion is NEVER re-run here, even on retry.
    """
    from agent import run_agents_in_parallel
    from embeddings import get_namespace
    from github import GithubException
    from google.genai.errors import ClientError, ServerError

    repo_url = f"https://github.com/{repo_full_name}"
    namespace = get_namespace(repo_url)
    pr_diff = _build_pr_diff_overview(files_changed)
    attempt = self.request.retries + 1

    logger.info(f"[review_task] Attempt {attempt}/{REVIEW_MAX_RETRIES} for PR#{pr_number}")

    # ── Run agents ────────────────────────────────────────────────────────────
    try:
        # asyncio.run() is safe here: Celery workers are sync, no running event loop
        final_review = asyncio.run(
            asyncio.wait_for(
                run_agents_in_parallel(repo_url=repo_url, pr_diff=pr_diff, ref=sha),
                timeout=300,  # 5 min hard cap — prevents infinite agent hangs
            )
        )
    except asyncio.TimeoutError:
        logger.error(f"[review_task] Agents timed out after 5 minutes for PR#{pr_number}")
        raise self.retry(
            exc=asyncio.TimeoutError("Agent timeout"),
            countdown=BASE_BACKOFF ** attempt,
        )
    except (ClientError, ServerError) as exc:
        err_str = str(exc)
        # Daily quota is non-recoverable — don't retry
        if "PerDay" in err_str or "per day" in err_str.lower() or "Daily" in err_str:
            logger.error(f"[review_task] Daily Gemini quota exhausted. Giving up on PR#{pr_number}.")
            return

        wait = BASE_BACKOFF ** attempt
        logger.warning(
            f"[review_task] Gemini error on attempt {attempt}: {exc}. "
            f"Retrying in {wait}s..."
        )
        raise self.retry(exc=exc, countdown=wait)
    except Exception as exc:
        wait = BASE_BACKOFF ** attempt
        logger.warning(f"[review_task] Unexpected error: {exc}. Retrying in {wait}s...")
        raise self.retry(exc=exc, countdown=wait)

    logger.info(f"[review_task] Agents completed. Posting comment to PR#{pr_number}...")

    # ── Post GitHub comment ───────────────────────────────────────────────────
    gh_client = _get_github_client(installation_id)
    if not gh_client:
        logger.error("[review_task] No GitHub client — cannot post comment.")
        return

    try:
        repo = gh_client.get_repo(repo_full_name)
        pull = repo.get_pull(pr_number)
        pull.create_issue_comment(final_review)
        logger.info(f"[review_task] ✅ Posted review comment on PR#{pr_number}")

        # Clear the idempotency key so re-opens of the same PR can be reviewed again
        idempotency_key = f"pr_review:{repo_full_name}:{pr_number}:{sha}"
        redis_client.delete(idempotency_key)

    except GithubException as exc:
        status = exc.status
        is_transient = status >= 500 or status == 429
        if is_transient:
            wait = BASE_BACKOFF ** attempt
            logger.warning(f"[review_task] GitHub API {status} — retrying in {wait}s...")
            raise self.retry(exc=exc, countdown=wait)
        else:
            logger.error(f"[review_task] GitHub API {status} non-retryable: {exc.data}")
    except Exception as exc:
        wait = BASE_BACKOFF ** attempt
        logger.warning(f"[review_task] GitHub connection error: {exc}. Retrying in {wait}s...")
        raise self.retry(exc=exc, countdown=wait)
