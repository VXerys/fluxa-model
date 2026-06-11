from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.voice_routes import router as voice_router

app = FastAPI(
    title="Fluxa Voice AI Backend",
    version="0.1.0",
    description="AI backend for Fluxa voice transaction parsing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Fluxa Voice AI Backend",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
