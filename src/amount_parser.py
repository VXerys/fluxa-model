"""Rule-based Indonesian/Sundanese money amount parser.

Designed for common Fluxa voice transaction utterances:
- lima rebu / lima rébu -> 5000
- lima belas rebu -> 15000
- tujuh ratus rébu -> 700000
- Rp1.500.000 -> 1500000
- hiji juta lima ratus rébu -> 1500000
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_UNITS = {
    "nol": 0,
    "satu": 1,
    "sa": 1,
    "hiji": 1,
    "dua": 2,
    "tilu": 3,
    "tiga": 3,
    "opat": 4,
    "empat": 4,
    "lima": 5,
    "genep": 6,
    "enam": 6,
    "tujuh": 7,
    "dalapan": 8,
    "delapan": 8,
    "salapan": 9,
    "sembilan": 9,
    "sapuluh": 10,
    "sepuluh": 10,
    "sabelas": 11,
    "sebelas": 11,
    "saratus": 100,
    "seratus": 100,
    "sarebu": 1000,
    "seribu": 1000,
    "sajuta": 1000000,
    "sejuta": 1000000,
}

_MULTIPLIERS = {
    "ribu": 1_000,
    "rebu": 1_000,
    "rb": 1_000,
    "k": 1_000,
    "juta": 1_000_000,
    "jt": 1_000_000,
    "m": 1_000_000,
}

_NUMBER_TOKENS = set(_UNITS) | {"belas", "puluh", "ratus"} | set(_MULTIPLIERS)


def _strip_accents(value: str) -> str:
    """Convert rébu -> rebu without disturbing normal ASCII text."""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_amount_text(text: str) -> str:
    if text is None:
        return ""

    value = _strip_accents(str(text).lower().strip())
    # Whisper sometimes transcribes "50 ribu" as "50W". Treat W as ribu in money context.
    value = re.sub(r"\b(\d+)\s*w\b", r"\1 ribu", value)
    value = value.replace("rupiah", " ")
    value = re.sub(r"[^\w\s.,]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_under_1000(tokens: list[str]) -> Optional[int]:
    """Parse number words below 1000, plus direct aliases like seribu/sajuta."""
    if not tokens:
        return None

    total = 0
    i = 0
    matched = False

    while i < len(tokens):
        tok = tokens[i]

        if tok in ("saratus", "seratus"):
            total += 100
            matched = True
            i += 1
            continue

        if tok in ("sarebu", "seribu"):
            total += 1000
            matched = True
            i += 1
            continue

        if tok in ("sajuta", "sejuta"):
            total += 1_000_000
            matched = True
            i += 1
            continue

        if tok in _UNITS:
            value = _UNITS[tok]

            if value >= 1000:
                total += value
                matched = True
                i += 1
                continue

            # dua belas / lima belas
            if i + 1 < len(tokens) and tokens[i + 1] == "belas" and value < 10:
                total += 10 + value
                matched = True
                i += 2
                continue

            # dua puluh lima
            if i + 1 < len(tokens) and tokens[i + 1] == "puluh" and value < 10:
                total += value * 10
                matched = True
                i += 2

                if i < len(tokens) and tokens[i] in _UNITS and _UNITS[tokens[i]] < 10:
                    total += _UNITS[tokens[i]]
                    i += 1

                continue

            # tujuh ratus
            if i + 1 < len(tokens) and tokens[i + 1] == "ratus" and value < 10:
                total += value * 100
                matched = True
                i += 2
                continue

            total += value
            matched = True

        elif tok == "puluh":
            total += 10
            matched = True

        elif tok == "ratus":
            total += 100
            matched = True

        i += 1

    return total if matched else None


def _parse_scaled_sequence(tokens: list[str]) -> Optional[int]:
    """Parse a contiguous sequence such as:
    - tujuh ratus rebu
    - hiji juta lima ratus rebu
    - dua ratus lima puluh rebu
    """
    if not tokens:
        return None

    total = 0
    current: list[str] = []
    used_multiplier = False

    for tok in tokens:
        if tok in _MULTIPLIERS:
            base = _parse_under_1000(current) if current else 1
            if base is None:
                base = 1

            total += base * _MULTIPLIERS[tok]
            current = []
            used_multiplier = True
        else:
            current.append(tok)

    leftover = _parse_under_1000(current)

    if total == 0 and leftover is not None:
        total = leftover

    if used_multiplier or total >= 1000:
        return total

    return None


def parse_amount(text: str) -> Optional[int]:
    normalized = _normalize_amount_text(text)
    if not normalized:
        return None

    # Rp35.000 / Rp1.500.000 / Rp750 000 / 1 500 000
    money_match = re.search(r"(?:rp\s*)?(\d{1,3}(?:[\s.,]\d{3})+)", normalized)
    if money_match:
        digits = re.sub(r"\D", "", money_match.group(1))
        if digits:
            return int(digits)

    # 35rb, 35k, 2jt, 2 juta
    compact_match = re.search(
        r"\b(\d+(?:[\.,]\d+)?)\s*(rb|ribu|rebu|k|jt|juta|m)\b",
        normalized,
    )
    if compact_match:
        number = float(compact_match.group(1).replace(",", "."))
        multiplier = _MULTIPLIERS[compact_match.group(2)]
        return int(number * multiplier)

    # Plain number such as 1000000.
    plain_match = re.search(r"\b\d{4,}\b", normalized)
    if plain_match:
        return int(plain_match.group(0))

    # Word-based amount. Extract contiguous number-token sequences.
    tokens = normalized.replace(".", " ").replace(",", " ").split()
    best_amount: Optional[int] = None
    sequence: list[str] = []

    for tok in tokens:
        if tok in _NUMBER_TOKENS:
            sequence.append(tok)
        else:
            if sequence:
                amount = _parse_scaled_sequence(sequence)
                if amount is not None and (best_amount is None or amount > best_amount):
                    best_amount = amount
                sequence = []

    if sequence:
        amount = _parse_scaled_sequence(sequence)
        if amount is not None and (best_amount is None or amount > best_amount):
            best_amount = amount

    return best_amount


def amount_exact_match(predicted: Optional[int], expected: Optional[int]) -> bool:
    if expected is None:
        return predicted is None
    return predicted == int(expected)