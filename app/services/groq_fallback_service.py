"""Groq LLM fallback service for correcting transaction fields.

This service is a *post-processor* that runs after the local ML pipeline.
It calls the Groq chat-completions API (OpenAI-compatible) to improve
title, description, category, wallet, and type — only when the local
parser output looks uncertain.

Groq MUST NOT override:
    - amount  (unless the local parser failed completely)
    - currency  (always IDR)

The service is **disabled by default** and requires both the feature flag
``ENABLE_GROQ_FALLBACK=true`` and a valid ``GROQ_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Allowed values (validation)
# ────────────────────────────────────────────────────────────────────────────

ALLOWED_CATEGORIES: set[str] = {
    "Makan & Minum",
    "Transportasi",
    "Belanja",
    "Tagihan & Utilitas",
    "Hiburan",
    "Kesehatan",
    "Gaji",
    "Freelance",
    "Transfer",
}

ALLOWED_TYPES: set[str] = {"expense", "income", "transfer"}

ALLOWED_WALLETS: dict[str, str] = {
    "bca": "BCA",
    "bri": "BRI",
    "bni": "BNI",
    "mandiri": "Mandiri",
    "dana": "DANA",
    "gopay": "GoPay",
    "go pay": "GoPay",
    "ovo": "OVO",
    "shopeepay": "ShopeePay",
    "shopee pay": "ShopeePay",
    "cash": "Cash",
    "tunai": "Cash",
}

# Canonical wallet names for direct match (Groq might return "BCA" directly).
_CANONICAL_WALLETS: set[str] = set(ALLOWED_WALLETS.values())

# ────────────────────────────────────────────────────────────────────────────
# Trigger-condition helpers
# ────────────────────────────────────────────────────────────────────────────

_AMOUNT_NOISE_PATTERN = re.compile(
    r"\b(?:rp\s*\d+|ribu|rebu|rb|\d{3,})\b", re.IGNORECASE,
)

_WALLET_WORDS: set[str] = {
    "bca", "bri", "bni", "mandiri",
    "dana", "gopay", "go pay", "ovo",
    "shopeepay", "shopee pay", "cash", "tunai",
}

_TRANSFER_PHRASES: set[str] = {
    "transfer", "kirim", "pindah", "pindahin", "tf",
}

_SUSPICIOUS_STT_WORDS: set[str] = {
    "hmm", "eh", "uh", "uhm", "hah", "kok", "loh",
    "aduh", "duh", "yah",
}


def _title_looks_bad(title: str | None) -> bool:
    """Return True if the title/description is empty, too short, or noisy."""
    if not title or len(title.strip()) < 3:
        return True
    return bool(_AMOUNT_NOISE_PATTERN.search(title))


def _transcript_contains_wallet(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in _WALLET_WORDS)


def _transcript_contains_transfer(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in _TRANSFER_PHRASES)


def _transcript_looks_noisy(normalized: str) -> bool:
    if len(normalized.strip()) < 5:
        return True
    tokens = normalized.lower().split()
    suspicious_count = sum(1 for t in tokens if t in _SUSPICIOUS_STT_WORDS)
    return suspicious_count >= 2 or (len(tokens) <= 3 and suspicious_count >= 1)


# ────────────────────────────────────────────────────────────────────────────
# Groq prompt
# ────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah asisten koreksi transaksi keuangan Indonesia.

Kamu akan menerima hasil parsing transaksi dari speech-to-text (Whisper) \
yang mungkin mengandung kesalahan pada judul, deskripsi, kategori, wallet, \
atau tipe transaksi.

Tugasmu:
1. Perbaiki **title** agar berisi nama item/tujuan transaksi yang bersih, \
   tanpa angka nominal, tanpa nama wallet. Gunakan huruf kapital di awal kata.
2. Buat **description** singkat 1 kalimat yang menjelaskan transaksi.
3. Perbaiki **category** jika tidak sesuai konteks. Kategori yang diizinkan: \
   Makan & Minum, Transportasi, Belanja, Tagihan & Utilitas, Hiburan, \
   Kesehatan, Gaji, Freelance, Transfer.
4. Perbaiki **type** jika tidak konsisten. Tipe yang diizinkan: \
   expense, income, transfer.
5. Perbaiki **wallet** jika terdeteksi di transkrip. Wallet yang diizinkan: \
   BCA, BRI, BNI, Mandiri, DANA, GoPay, OVO, ShopeePay, Cash.

Aturan penting:
- Jangan mengubah nominal/amount.
- Jika suatu field sudah benar, kirim null untuk field tersebut.
- Jika tidak yakin, kirim null.
- Jawab HANYA dengan JSON valid, tanpa teks lain.

Format respons (JSON saja):
{
  "title": "string atau null",
  "description": "string atau null",
  "type": "expense|income|transfer atau null",
  "category": "kategori yang diizinkan atau null",
  "wallet": "wallet yang diizinkan atau null",
  "reason": "alasan singkat koreksi"
}\
"""


def _build_user_message(local_result: dict[str, Any]) -> str:
    transcript = local_result.get("transcript", {})
    transaction = local_result.get("transaction", {})
    warnings = local_result.get("warnings", [])

    return json.dumps(
        {
            "raw_transcript": transcript.get("raw", ""),
            "normalized_transcript": transcript.get("normalized", ""),
            "local_amount": transaction.get("amount"),
            "local_type": transaction.get("type", ""),
            "local_category": transaction.get("category", ""),
            "local_wallet": transaction.get("wallet"),
            "local_title": transaction.get("title"),
            "local_description": transaction.get("description"),
            "warnings": warnings,
        },
        ensure_ascii=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ────────────────────────────────────────────────────────────────────────────

def _validate_wallet(value: str | None) -> str | None:
    """Return canonical wallet name or None if invalid."""
    if value is None:
        return None
    if value in _CANONICAL_WALLETS:
        return value
    return ALLOWED_WALLETS.get(value.lower().strip())


def _validate_category(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value in ALLOWED_CATEGORIES else None


def _validate_type(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower().strip()
    return lowered if lowered in ALLOWED_TYPES else None


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────

class GroqFallbackService:
    """Groq LLM fallback post-processor for transaction correction."""

    def __init__(self) -> None:
        self._enabled = os.getenv("ENABLE_GROQ_FALLBACK", "false").lower() == "true"
        self._api_key = os.getenv("GROQ_API_KEY", "")
        self._base_url = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self._model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._timeout = int(os.getenv("GROQ_TIMEOUT_SECONDS", "8"))

    # ── public ──────────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled and bool(self._api_key)

    def should_trigger(self, local_result: dict[str, Any]) -> bool:
        """Evaluate whether the local result warrants a Groq fallback call."""
        if not self.is_enabled:
            return False

        warnings = local_result.get("warnings", [])
        transcript = local_result.get("transcript", {})
        transaction = local_result.get("transaction", {})

        raw_text = transcript.get("raw", "")
        normalized = transcript.get("normalized", "")
        local_type = transaction.get("type", "")
        local_category = transaction.get("category", "")
        local_wallet = transaction.get("wallet")
        local_title = transaction.get("title") or transaction.get("description")

        # 1. Warnings present
        if warnings:
            return True

        # 2. Title is bad
        if _title_looks_bad(local_title):
            return True

        # 3. Category "Transfer" but type is expense/income and no transfer phrase
        if (
            local_category == "Transfer"
            and local_type in ("expense", "income")
            and not _transcript_contains_transfer(normalized or raw_text)
        ):
            return True

        # 4. Wallet is null but transcript contains wallet words
        if local_wallet is None and _transcript_contains_wallet(
            normalized or raw_text
        ):
            return True

        # 5. Transcript looks noisy
        if _transcript_looks_noisy(normalized or raw_text):
            return True

        # 6. Low confidence (if present)
        confidence = transcript.get("confidence")
        if confidence is not None and confidence < 0.5:
            return True

        return False

    def call_groq(self, local_result: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Call Groq chat completions and return parsed JSON response.

        Returns None on any failure (timeout, HTTP error, bad JSON, etc.).
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(local_result)},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()

            body = response.json()
            content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Strip markdown code fences if present.
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                logger.warning("Groq returned non-dict JSON: %s", type(parsed))
                return None

            return parsed

        except httpx.TimeoutException:
            logger.warning("Groq API call timed out after %ds", self._timeout)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning("Groq API HTTP error: %s", exc.response.status_code)
            return None
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Groq fallback failed: %s", exc)
            return None

    def apply_corrections(
        self,
        local_result: dict[str, Any],
        groq_response: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge validated Groq corrections into the local result.

        Returns a *new* dict (the original is not mutated).
        """
        result = {
            "transcript": dict(local_result.get("transcript", {})),
            "transaction": dict(local_result.get("transaction", {})),
            "classification": dict(local_result.get("classification", {})),
            "warnings": list(local_result.get("warnings", [])),
        }

        tx = result["transaction"]
        applied = False

        # Title
        groq_title = groq_response.get("title")
        if groq_title and isinstance(groq_title, str) and groq_title.strip():
            tx["title"] = groq_title.strip()
            applied = True

        # Description
        groq_desc = groq_response.get("description")
        if groq_desc and isinstance(groq_desc, str) and groq_desc.strip():
            tx["description"] = groq_desc.strip()
            applied = True

        # Category
        groq_category = _validate_category(groq_response.get("category"))
        if groq_category and groq_category != tx.get("category"):
            tx["category"] = groq_category
            result["classification"]["category"] = groq_category
            applied = True

        # Type
        groq_type = _validate_type(groq_response.get("type"))
        if groq_type and groq_type != tx.get("type"):
            tx["type"] = groq_type
            result["classification"]["resolved_type"] = groq_type
            applied = True

        # Wallet
        groq_wallet = _validate_wallet(groq_response.get("wallet"))
        if groq_wallet and groq_wallet != tx.get("wallet"):
            tx["wallet"] = groq_wallet
            applied = True

        if applied:
            result["warnings"].append("groq_fallback_used")

        return result

    def maybe_apply(self, local_result: dict[str, Any]) -> dict[str, Any]:
        """Main entry point: check trigger, call Groq, apply corrections.

        Always returns a valid result dict (original or corrected).
        """
        if not self.should_trigger(local_result):
            return local_result

        logger.info("Groq fallback triggered — calling Groq API")
        groq_response = self.call_groq(local_result)

        if groq_response is None:
            # Groq failed — keep local result, add warning.
            result = {
                "transcript": dict(local_result.get("transcript", {})),
                "transaction": dict(local_result.get("transaction", {})),
                "classification": dict(local_result.get("classification", {})),
                "warnings": list(local_result.get("warnings", [])),
            }
            result["warnings"].append("groq_fallback_failed")
            return result

        return self.apply_corrections(local_result, groq_response)


# Module-level singleton
groq_fallback_service = GroqFallbackService()
