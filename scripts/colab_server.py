# ═══════════════════════════════════════════════════════════════════
# Ceiling AI — Google Colab Server Launcher
# Run this in a Colab cell to expose the FastAPI backend via ngrok
#
# Colab free tier: ~12GB RAM, T4 GPU available — perfect for viva demo
# ═══════════════════════════════════════════════════════════════════
#
# ── CELL 1: Install dependencies ────────────────────────────────────
# !pip install pyngrok fastapi uvicorn[standard] python-multipart \
#     pydantic==2.9.2 pydantic-settings==2.5.2 "pydantic[email]" \
#     python-jose[cryptography] passlib[bcrypt] bcrypt==4.0.1 \
#     sqlalchemy==2.0.36 psycopg2-binary==2.9.9 \
#     torch==2.4.1 torchvision==0.19.1 \
#     segmentation-models-pytorch==0.5.0 \
#     opencv-python-headless==4.10.0.84 Pillow==10.4.0 numpy
#
# ── CELL 2: Clone your repo ──────────────────────────────────────────
# !git clone https://github.com/KumudithaRanasinghe/Ceiling_segmentation.git
# %cd Ceiling_segmentation
# Upload your .pth files to:  models/V1/best_model_optimized.pth
#                              models/V2/best_model_optimized_v2.pth
#
# ── CELL 3: Set environment variables ────────────────────────────────
import os
os.environ["DATABASE_URL"]              = "postgresql+psycopg2://neondb_owner:npg_QEkGUleMj3q0@ep-delicate-firefly-azwlyq0j-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
os.environ["SECRET_KEY"]               = "b70a6a585e2ee2b88e4010764c6b40b9eb615fd0235ffa2106feb9036aab64a9"
os.environ["DEBUG"]                    = "false"
os.environ["DEVICE"]                   = "cuda"   # or "cpu" if no GPU
os.environ["SEGMENTER_MODEL_PATH"]     = "models/V1/best_model_optimized.pth"
os.environ["V2_SEGMENTER_MODEL_PATH"]  = "models/V2/best_model_optimized_v2.pth"
os.environ["V2_NUM_CLASSES"]           = "2"
os.environ["NUM_CLASSES"]              = "4"
os.environ["ALLOWED_ORIGINS"]          = '["*"]'

#
# ── CELL 4: Start server + ngrok tunnel ──────────────────────────────
import subprocess, threading
# pyrefly: ignore [missing-import]
from pyngrok import ngrok, conf

# Set your ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"
conf.get_default().auth_token = NGROK_TOKEN

# Start FastAPI in a background thread
def run_server():
    subprocess.run([
        "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1"
    ])

t = threading.Thread(target=run_server, daemon=True)
t.start()

import time; time.sleep(5)  # Wait for startup

# Open ngrok tunnel
public_url = ngrok.connect(8000).public_url
print(f"\n{'='*60}")
print(f"  Ceiling AI is LIVE at:")
print(f"  {public_url}/api/v1/docs")
print(f"{'='*60}\n")
print(f"  Health check: {public_url}/api/v1/health/live")
print(f"  Update Flutter base URL to: {public_url}")
