from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


class SttService:
    """Speech-to-text service using faster-whisper.

    Default config is CPU-safe for Camber/Jupyter:
    - model_size: base
    - device: cpu
    - compute_type: int8

    Later, change FLUXA_WHISPER_MODEL_SIZE to "small" when the environment is strong enough.
    """

    def __init__(self) -> None:
        self.model_size = os.getenv("FLUXA_WHISPER_MODEL_SIZE", "base")
        self.device = os.getenv("FLUXA_WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("FLUXA_WHISPER_COMPUTE_TYPE", "int8")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            str(audio_path),
            language="id",
            beam_size=5,
            vad_filter=True,
        )

        text_parts: list[str] = []
        for segment in segments:
            if segment.text:
                text_parts.append(segment.text.strip())

        transcript = " ".join(text_parts).strip()

        return {
            "text": transcript,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        }


stt_service = SttService()
