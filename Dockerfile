# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Keep Python fast and predictable in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for typical Python builds (remove if unnecessary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt* /app/
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the rest of the application
COPY . /app

# Run as non-root for better security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Override this in docker-compose.yml as needed
CMD ["python", "main.py"]
