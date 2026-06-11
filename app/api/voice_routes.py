from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.voice_transaction_schema import (
    ParseTextRequest,
    VoiceTransactionResponse,
)
from app.services.stt_service import stt_service
from app.services.transaction_parser_service import transaction_parser_service

router = APIRouter(prefix="/api/v1/voice", tags=["Voice Transaction"])


@router.post("/parse-text", response_model=VoiceTransactionResponse)
def parse_text(request: ParseTextRequest) -> VoiceTransactionResponse:
    try:
        result = transaction_parser_service.parse_text(request.text)
        return VoiceTransactionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse transaction text: {str(exc)}",
        ) from exc


@router.post("/parse", response_model=VoiceTransactionResponse)
async def parse_audio(file: UploadFile = File(...)) -> VoiceTransactionResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_audio_path = Path(tmp.name)

        stt_result = stt_service.transcribe(temp_audio_path)

        transcript_text = stt_result.get("text") or ""
        if not transcript_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Whisper could not detect any transcript from the audio.",
            )

        result = transaction_parser_service.parse_text(transcript_text)

        result["transcript"]["language_hint"] = stt_result.get("language")
        result["transcript"]["confidence"] = stt_result.get("language_probability")

        return VoiceTransactionResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse transaction audio: {str(exc)}",
        ) from exc
    finally:
        try:
            if "temp_audio_path" in locals() and temp_audio_path.exists():
                temp_audio_path.unlink()
        except Exception:
            pass
