---
title: Ceiling AI
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Ceiling AI — Ceiling Material Segmentation API

AI-powered ceiling material estimation from architectural floor plan images.

## Features
- Upload a ceiling/floor plan image → get a segmentation mask + material area quantities
- V1 model: Multi-class room segmentation (UNet++ EfficientNet-B4)
- V2 model: Binary inside-floor vs background detection (UNet++ EfficientNet-B4)
- JWT authentication (register → login → segment)
- Admin dashboard with job management

## Quick Start

1. Open the interactive API docs: `/api/v1/docs`
2. Register: `POST /api/v1/auth/register`
3. Login: `POST /api/v1/auth/login` → copy the `access_token`
4. Click **Authorize** in the docs UI → paste the token
5. Upload an image: `POST /api/v1/segment/`

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/auth/register` | No | Create account |
| POST | `/api/v1/auth/login` | No | Get JWT token |
| POST | `/api/v1/segment/` | Yes | Run segmentation |
| GET | `/api/v1/health/live` | No | Liveness probe |
| GET | `/api/v1/health/ready` | No | Readiness probe (checks models) |
| GET | `/api/v1/docs` | No | Swagger UI |

## Model Architecture
- **Encoder**: EfficientNet-B4
- **Decoder**: UNet++ with SCSE attention
- **Input**: 512×512 RGB
- **Output**: Multi-class segmentation mask + per-class area in m²

## Thesis
Developed as part of a BSc/MSc research project on automated ceiling material estimation from architectural plans.
