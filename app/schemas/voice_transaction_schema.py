from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TranscriptResponse(BaseModel):
    raw: str
    normalized: str
    language_hint: Optional[str] = None
    confidence: Optional[float] = None


class TransactionResponse(BaseModel):
    type: str
    amount: Optional[int]
    category: str
    wallet: Optional[str] = None
    description: Optional[str] = None
    currency: str = "IDR"


class ClassificationResponse(BaseModel):
    raw_type: str
    resolved_type: str
    category: str


class VoiceTransactionResponse(BaseModel):
    transcript: TranscriptResponse
    transaction: TransactionResponse
    classification: ClassificationResponse
    warnings: list[str] = []
