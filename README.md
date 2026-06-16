---
title: Fluxa Voice AI Backend
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Fluxa Voice AI Backend

FastAPI backend for Fluxa voice transaction parsing.

## Endpoints

- `GET /health`
- `POST /api/v1/voice/parse-text`
- `POST /api/v1/voice/parse`

## Pipeline

Audio → faster-whisper → text normalization → transaction classifier → amount parser → title extractor → transaction JSON.

### Groq Fallback (Optional)

When enabled, a Groq LLM post-processor can correct **title**, **description**, **category**, **wallet**, and **type** fields when the local parser output looks uncertain.

Groq is called **only** when one or more of these conditions are true:
- The `warnings` list is not empty
- The title is empty, too short, or contains amount noise
- Category/type combination is inconsistent
- Wallet is missing but transcript mentions a wallet name
- The transcript looks noisy or contains STT artifacts
- STT confidence is low

Groq **never** overrides the `amount` or `currency` fields.

#### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENABLE_GROQ_FALLBACK` | `false` | Set to `true` to enable Groq fallback |
| `GROQ_API_KEY` | *(empty)* | Your Groq API key (required when enabled) |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model for the fallback |
| `GROQ_TIMEOUT_SECONDS` | `8` | Timeout per Groq API call |

#### Warnings

- `"groq_fallback_used"` — Groq corrections were applied
- `"groq_fallback_failed"` — Groq was attempted but failed (local result kept)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # edit with your values
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Testing

```bash
python -m pytest tests/ -v
```
