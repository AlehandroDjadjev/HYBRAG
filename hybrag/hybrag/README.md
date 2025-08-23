# HYBRAG – Technical Construction Image Query System

This repo contains a Django REST API (backend) and a minimal Next.js app (frontend) for semantic image search using SigLIP embeddings and a Pinecone vector database.

## Components

- Backend: Django + DRF
  - Endpoints:
    - `POST /api/images` – upload image + metadata, embed with SigLIP, upsert into Pinecone
    - `GET /api/search` – text or reference-image search with optional metadata filters
  - Local file storage for images under `MEDIA/` (development)
  - SQLite for canonical records (development)
- Vector DB: Pinecone (managed)
- Embeddings: SigLIP via Hugging Face Transformers (`google/siglip-base-patch16-224`)
- Frontend: Next.js (minimal UI to ingest and search)

## Project layout

```
hybrag/
  hybrag/
    backend/
      images/               # Django app
        embeddings/         # SigLIP service
        vector/             # Pinecone client wrapper
      server/               # Django project
      requirements.txt
      manage.py
      .env.example
    frontend/               # Next.js app
      pages/
      package.json
      next.config.js
      tsconfig.json
      env.sample.txt
    README.md
```

## Quickstart (Development)

Prerequisites:
- Python 3.10+
- Node 18+
- Pinecone account, API key, and an index (dimension = SigLIP dim, cosine)

1) Backend setup

```bash
cd hybrag/hybrag/backend
copy .env.example .env   # or cp on mac/linux
# Fill PINECONE_API_KEY and either PINECONE_ENV (classic) or PINECONE_HOST (serverless)
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

API at `http://127.0.0.1:8000`.

2) Frontend setup

```bash
cd hybrag/hybrag/frontend
copy env.sample.txt .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## Pinecone configuration

- Classic projects: set `PINECONE_API_KEY`, `PINECONE_ENV` (e.g., `us-east1-gcp` or `us-east-1`), and `PINECONE_INDEX`.
- Serverless projects: set `PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_HOST` to your index host URL shown in the console (format like `https://<index>-<project>.svc.<region>-<cloud>.pinecone.io`). Leave `PINECONE_ENV` blank.

If you see DNS errors for `controller.<region>.pinecone.io`, use the serverless `PINECONE_HOST` path to bypass control-plane lookups.

## API

### POST /api/images
multipart form-data fields:
- `file` (required): image file
- `building` (required): string
- `shot_date` (required): `YYYY-MM-DD`
- `notes` (optional): string

Response: `{ id, image_url }`

### GET /api/search
Query params:
- `q` (optional): text query
- `query_image_id` (optional): UUID of an already uploaded image to use as the query
- `building` (optional): exact filter
- `date_from`, `date_to` (optional): `YYYY-MM-DD`
- `k` (optional): top-k (default 10)

Response: `{ results: [ { id, score, image_url, building, shot_date, notes } ] }`

## What API keys/services you need

- Pinecone: API key + environment or index host + index name.
- Storage: local `MEDIA/` directory for dev. For prod, use S3/GCS and store URLs in metadata.
- Embeddings: SigLIP from Hugging Face model hub (no key required unless gated/private).

Backend `.env` example (classic):
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=us-east-1
PINECONE_INDEX=construction-images
SIGLIP_MODEL_NAME=google/siglip-base-patch16-224
DEVICE=auto
SECRET_KEY=dev-secret-change-me
DEBUG=true
ALLOWED_HOSTS=*
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
MEDIA_ROOT=media
MEDIA_URL=/media/
```

Backend `.env` example (serverless):
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=construction-images
PINECONE_HOST=https://<index>-<project>.svc.us-east-1-aws.pinecone.io
# Leave PINECONE_ENV blank
```

Frontend `.env.local`:
```
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

## Notes on date filtering

We store both `shot_date` (`YYYY-MM-DD`) and `shot_ymd` (e.g., `20230712`) in the vector payload for efficient range filters with Pinecone.


