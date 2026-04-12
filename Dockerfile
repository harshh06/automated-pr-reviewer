# Use the official lightweight Python image
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY backend/ .

# Run the web service on container startup using Uvicorn.
# Cloud Run injects the PORT environment variable automatically (default 8080).
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}