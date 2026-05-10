# syntax=docker/dockerfile:1.6
#
# Single Dockerfile (не Dockerfile.api как в design doc PR 14) — упрощение:
# после PR 13 Streamlit decommissioned, нет нужды в split. Один Dockerfile
# поддерживает HF Spaces (PORT=7860) и local docker-compose (PORT=8000 override).
#
# Multi-stage build for Steel AI MVP — FastAPI + vanilla JS UI.
# Streamlit was decommissioned in PR 13 of the FastAPI migration; PR 14
# replaces the single-stage stop-gap with a proper multi-stage layout.
#
# Stage 1 (builder): installs Python deps inside a venv. Carries build-essential
#   for native wheels (XGBoost, sklearn) but stays out of the runtime image.
# Stage 2 (runtime):  slim image without compilers. Drops the venv in, copies
#   only project source, switches to a non-root `steel` user.

# ---------- Stage 1: builder ----------
FROM python:3.11-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# Non-root user per OCI / HF Spaces security best practices.
RUN useradd --create-home --shell /bin/bash steel

# Pre-built venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Project files. We deliberately do NOT `COPY . .` — that would pull in
# tests/, models/, data/*.parquet, decision_log/*.db, .git/, etc.
COPY --chown=steel:steel app/ ./app/
COPY --chown=steel:steel data/ ./data/
COPY --chown=steel:steel decision_log/ ./decision_log/
COPY --chown=steel:steel pattern_library/ ./pattern_library/
COPY --chown=steel:steel scripts/ ./scripts/
COPY --chown=steel:steel docs/ ./docs/
COPY --chown=steel:steel requirements.txt ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER steel

# HF Spaces (Docker SDK) listens on 7860 by default; PORT env allows override
# for local docker-compose (mapped to 8000 there).
EXPOSE 7860
ENV PORT=7860
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT}"]
