"""End-to-end inference pipeline for Fluxa voice transaction parser.

Pipeline:
text
-> type classifier
-> category classifier
-> transaction type resolver
-> amount parser
-> title extractor
-> transaction JSON draft
-> Groq fallback (optional, when local result is uncertain)

The resolver is important because some category labels imply a fixed transaction type:
- Gaji, Freelance -> income
- Transfer -> transfer
- Other categories -> expense or model prediction
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from src.amount_parser import parse_amount
from src.date_parser import parse_date
from src.text_normalizer import normalize_text
from src.title_extractor import extract_title
from src.pipeline_logger import pipeline_logger
from app.services.groq_fallback_service import groq_fallback_service


INCOME_CATEGORIES = {"Gaji", "Freelance"}
TRANSFER_CATEGORIES = {"Transfer"}

CATEGORY_KEYWORDS = {
    "Tagihan & Utilitas": [
        "wifi",
        "wi-fi",
        "internet",
        "kuota",
        "listrik",
        "token",
        "pulsa",
        "kos",
        "kontrakan",
    ],
    "Transportasi": [
        "bensin",
        "parkir",
        "ojek",
        "ojol",
        "angkot",
        "grab",
        "gojek",
        "transport",
    ],
    "Makan & Minum": [
        "kopi",
        "nasi",
        "cilok",
        "seblak",
        "makan",
        "jajan",
        "sangu",
        "minum",
    ],
    "Kesehatan": [
        "obat",
        "apotik",
        "apotek",
        "dokter",
        "klinik",
        "vitamin",
    ],
}


@dataclass
class TransactionDraft:
    type: str
    amount: Optional[int]
    category: str
    wallet: Optional[str]
    title: Optional[str]
    description: Optional[str]
    currency: str = "IDR"
    date: Optional[str] = None
    subcategory: Optional[str] = None


@dataclass
class ClassificationResult:
    raw_type: str
    resolved_type: str
    category: str


@dataclass
class VoiceTransactionResult:
    transcript: dict[str, Any]
    transaction: TransactionDraft
    classification: ClassificationResult
    warnings: list[str]


def resolve_category_by_keywords(text: str, pred_category: str) -> tuple[str, str | None]:
    """Resolve obvious category conflicts using finance-domain keywords.

    Example:
    text = "mayar wifi rp 300000"
    model category = Gaji
    resolved category = Tagihan & Utilitas
    """
    lowered = text.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            if pred_category != category:
                return category, f"category_resolved_by_keyword:{pred_category}->{category}"
            return pred_category, None

    return pred_category, None


def resolve_transaction_type(pred_type: str, pred_category: str) -> str:
    """Resolve inconsistent model outputs using finance domain rules.

    Rules:
    - Gaji, Freelance -> income
    - Transfer -> transfer
    - All other categories -> expense
    """
    if pred_category in INCOME_CATEGORIES:
        return "income"

    if pred_category in TRANSFER_CATEGORIES:
        return "transfer"

    return "expense"


def infer_transaction(
    text: str,
    type_model: Any,
    category_model: Any,
    wallet_model: Any | None = None,
) -> dict[str, Any]:
    """Run complete transaction inference from raw text.

    Args:
        text: Raw user text / Whisper transcript.
        type_model: Trained type classifier.
        category_model: Trained category classifier.
        wallet_model: Optional wallet classifier.

    Returns:
        Dict compatible with Fluxa backend response draft.
    """
    t_start = time.perf_counter()

    # Step 0: Normalize
    normalized = normalize_text(text)
    pipeline_logger.log_raw_input(text)

    # Step 1: Date — sequential tuple-returning parse
    t0 = time.perf_counter()
    parsed_date, text_after_date = parse_date(normalized)
    pipeline_logger.log_after_date(parsed_date, text_after_date, (time.perf_counter() - t0) * 1000)

    # Step 2: Amount — consumes date-cleaned text
    t0 = time.perf_counter()
    pred_amount, text_after_amount = parse_amount(text_after_date)
    pipeline_logger.log_after_amount(pred_amount, text_after_amount, (time.perf_counter() - t0) * 1000)

    # Step 3: Title + Description — consume amount-cleaned text
    t0 = time.perf_counter()
    local_title = extract_title(text_after_amount)
    
    # Description = full normalized transcript (clean version without typos)
    # This provides context about the full transaction in clean Indonesian
    local_desc = normalized if normalized else None
    
    pipeline_logger.log_after_title_desc(local_title, local_desc, (time.perf_counter() - t0) * 1000)

    # Step 4: Classification
    raw_pred_type = str(type_model.predict([text])[0])
    raw_pred_category = str(category_model.predict([text])[0])
    pred_category, category_warning = resolve_category_by_keywords(
        normalized,
        raw_pred_category,
    )
    resolved_type = resolve_transaction_type(raw_pred_type, pred_category)
    t0 = time.perf_counter()
    pipeline_logger.log_final_prediction(pred_category, resolved_type, (time.perf_counter() - t0) * 1000)

    pred_wallet: Optional[str] = "Cash"
    # Bypassed other wallet options for now
    # if wallet_model is not None:
    #     wallet_prediction = wallet_model.predict([text])[0]
    #     if wallet_prediction is not None and str(wallet_prediction).lower() not in {"none", "null", "nan"}:
    #         pred_wallet = str(wallet_prediction)

    warnings: list[str] = []

    if category_warning is not None:
        warnings.append(category_warning)

    if pred_amount is None:
        warnings.append("amount_not_detected")

    if pred_amount is not None and pred_amount > 50_000_000:
        warnings.append("amount_unusually_large")

    if raw_pred_type != resolved_type:
        warnings.append(
            f"type_resolved_by_category:{raw_pred_type}->{resolved_type}"
        )

    # Build parser_hints for Groq fallback suppression
    parser_hints = {
        "title_extracted": bool(local_title and len(local_title.strip()) > 3),
        "date_extracted": parsed_date is not None,
        "description_extracted": bool(local_desc),
    }

    transaction = TransactionDraft(
        type=resolved_type,
        amount=pred_amount,
        category=pred_category,
        wallet=pred_wallet,
        title=local_title if local_title else None,
        description=local_desc if local_desc else None,
        currency="IDR",
        date=parsed_date,
        subcategory=None,
    )

    classification = ClassificationResult(
        raw_type=raw_pred_type,
        resolved_type=resolved_type,
        category=pred_category,
    )

    local_result = {
        "transcript": {
            "raw": text,
            "normalized": normalized,
            "language_hint": None,
            "confidence": None,
        },
        "transaction": asdict(transaction),
        "classification": asdict(classification),
        "warnings": warnings,
        "parser_hints": parser_hints,
    }

    # --- Groq fallback (optional post-processor) ---
    result = groq_fallback_service.maybe_apply(local_result)

    total_ms = (time.perf_counter() - t_start) * 1000
    pipeline_logger.log_duration(total_ms)
    if total_ms > 500:
        pipeline_logger.log_performance_warning(total_ms)

    return result