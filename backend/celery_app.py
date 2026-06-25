"""
celery_app.py
Celery application config. All tasks are defined in tasks.py.

Key settings explained:
  task_acks_late          — task is acknowledged AFTER it completes, not when picked up.
                            If the worker dies mid-task, the job goes back to the queue.
  task_reject_on_worker_lost — pairs with acks_late: explicitly re-queues on worker crash.
  worker_prefetch_multiplier — each worker fetches only 1 job at a time. Prevents one worker
                               hogging all jobs while others are idle.
"""
import os
import sys
from celery import Celery
from dotenv import load_dotenv

# Celery starts as a standalone process — it doesn't go through main.py.
# We must explicitly load .env here so REDIS_URL and other vars are available.
load_dotenv()

# Ensure the backend directory is on sys.path so forked worker subprocesses
# can import project modules (ingestion, embeddings, tools, etc.)
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "pr_reviewer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Reliability settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    
    # ── Connection Limits for Free Tier Redis ──
    # The Redis Labs free tier only allows 30 concurrent connections.
    # Celery's default broker pool is 10. We explicitly set it to 10 here.
    broker_pool_limit=10,
    redis_max_connections=20,
)
