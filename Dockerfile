# ══════════════════════════════════════════════════════════════════════════════
# Ceiling AI — Hugging Face Spaces Docker Image
# Base: python:3.10-slim | Port: 7860 | Worker: 1 (both models in-process)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── System packages ────────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0 required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ────────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Install CPU-only PyTorch FIRST (avoids pulling the 2 GB CUDA wheel) ───────
RUN pip install --no-cache-dir \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# ── Install the rest of the application dependencies ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Non-root user (Hugging Face Spaces requires UID 1000) ─────────────────────
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/results /app/models/V1 /app/models/V2 \
    && chown -R appuser:appuser /app

# ── Copy application source ────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

USER appuser

# ── Expose the port Spaces expects ────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/health/live || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# ONE worker only: each worker loads both UNet++ models (~400-500 MB together).
# Rate limiter is in-memory so multiple workers would give independent counters.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
