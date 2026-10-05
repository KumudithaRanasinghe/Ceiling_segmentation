# Ceiling AI — HF Spaces Deployment Runbook

## What's been done locally
- [x] `ceiling_ai.db` removed from git tracking
- [x] `Dockerfile` created (python:3.10-slim, CPU torch, UID 1000, port 7860)
- [x] `.dockerignore` created
- [x] `requirements.txt` — psycopg2-binary added, pymysql removed, test packages split out
- [x] `requirements.dev.txt` — test packages (pytest, httpx, pytest-asyncio)
- [x] `app/infrastructure/database/session.py` — Postgres pool block added for Neon
- [x] `app/core/config.py` — model paths updated to `models/V1/` and `models/V2/`
- [x] `README.md` — HF Space YAML header added (sdk: docker, app_port: 7860)
- [x] `.gitattributes` — LFS tracking rules for *.pth
- [x] `models/V1/best_model_optimized.pth` — ready locally (235 MB)
- [x] `models/V2/best_model_optimized_v2.pth` — ready locally (235 MB)

## Next: Push to HF Space (run these in terminal)

### 1. Install git-lfs (needs sudo password in terminal)
```bash
sudo apt-get install -y git-lfs
cd /home/kumuditha/Desktop/Ceiling_segmentation
git lfs install
```

### 2. Create the Space on huggingface.co
Go to: https://huggingface.co/new-space
- Name: ceiling-ai
- SDK: Docker
- Hardware: CPU Basic (Free)
- Visibility: Private (for now)
- Click Create Space

### 3. Add the Space remote
```bash
cd /home/kumuditha/Desktop/Ceiling_segmentation
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/ceiling-ai
# e.g: git remote add space https://huggingface.co/spaces/kumuditha/ceiling-ai
```

### 4. Force-add the model weights via LFS (they're gitignored normally)
```bash
git add -f models/V1/best_model_optimized.pth
git add -f models/V2/best_model_optimized_v2.pth
git commit -m "feat: add model weights via Git LFS"
```

### 5. Push to the Space (will upload ~470 MB of LFS objects)
```bash
git push space test/v1:main
```
This will take a while — git LFS uploads both .pth files.

## Set these Secrets in Space Settings → Variables and secrets

| Key | Value | Type |
|---|---|---|
| SECRET_KEY | b70a6a585e2ee2b88e4010764c6b40b9eb615fd0235ffa2106feb9036aab64a9 | Secret |
| DEBUG | false | Variable |
| DATABASE_URL | postgresql+psycopg2://neondb_owner:npg_QEkGUleMj3q0@ep-delicate-firefly-azwlyq0j-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require | Secret |
| SEGMENTER_MODEL_PATH | models/V1/best_model_optimized.pth | Variable |
| V2_SEGMENTER_MODEL_PATH | models/V2/best_model_optimized_v2.pth | Variable |
| V2_NUM_CLASSES | 4 | Variable |
| DEVICE | cpu | Variable |
| ALLOWED_ORIGINS | ["*"] | Variable |

## Verify after deployment
```
https://YOUR_HF_USERNAME-ceiling-ai.hf.space/api/v1/docs
https://YOUR_HF_USERNAME-ceiling-ai.hf.space/api/v1/health/live
https://YOUR_HF_USERNAME-ceiling-ai.hf.space/api/v1/health/ready
```
Watch Logs tab for:
  "V1 segmenter ready in ... ms on cpu"
  "V2 segmenter ready in ... ms on cpu"
