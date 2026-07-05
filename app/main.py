from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.voice_routes import router as voice_router

# Configure logging for production (Hugging Face)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set logger for the app
logger = logging.getLogger(__name__)

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
    logger.info("Root endpoint accessed")
    return {
        "status": "ok",
        "service": "Fluxa Voice AI Backend",
    }


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("Health check accessed")
    return {"status": "healthy"}
