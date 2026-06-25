"""
redis_client.py
Shared Redis connection. Import `redis_client` from here everywhere.
Reads REDIS_URL from environment (set in .env for local, Cloud Run env var in prod).
"""
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Restrict the connection pool to 5 connections max to prevent 
# "ERR max number of clients reached" on the Redis Labs free tier.
redis_client = redis.from_url(REDIS_URL, decode_responses=True, max_connections=5)
