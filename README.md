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

Audio → faster-whisper → text normalization → transaction classifier → amount parser → transaction JSON.
