# Use the official lightweight Python image
# 3.11+ required for `int | None` union syntax used in tasks.py
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# PYTHONPATH ensures Celery worker subprocesses can import project modules
ENV PYTHONPATH /app

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY backend/ .

# startup.sh runs both the FastAPI web server and the Celery worker in the same container.
# The '&' sends uvicorn to the background; Celery runs in the foreground (holds the container alive).
# Both share the same environment variables (REDIS_URL, GEMINI_API_KEY, etc.) injected by Cloud Run.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} & celery -A celery_app worker --loglevel=info --concurrency=2