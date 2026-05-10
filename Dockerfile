FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# FastAPI + vanilla JS UI (Streamlit decommissioned in PR 13).
# A dedicated Dockerfile.api with multi-stage build + non-root user lands in
# PR 14 of the migration; this stop-gap keeps `docker-compose up` working.
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
