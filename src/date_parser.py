"""Rule-based relative date parser for Indonesian/Sundanese voice transcripts.

Resolves relative time expressions that commonly appear in voice-based
financial transactions:

    - "kemarin" / "kamari"              → yesterday
    - "kemarin lusa" / "mangkukna"      → day before yesterday
    - "hari ini" / "ayeuna" / "sekarang"→ today
    - "besok" / "isukan"                → tomorrow
    - "senin kemarin"                   → most recent past Monday
    - "poe senen kemari"                → most recent past Monday (Sundanese)
    - "hari jumat kemarin"              → most recent past Friday
    - "tanggal 4 juni" / "4 juni"       → absolute date in current/nearest year
    - "tanggal 15" / "tgl 15"           → 15th of current month

All outputs are ISO-8601 date strings (``YYYY-MM-DD``).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Optional

# Placeholder logger for validation errors
pipeline_logger = logging.getLogger("fluxa.pipeline")

# ────────────────────────────────────────────────────────────────────────────
# Day-name mapping (Indonesian and Sundanese → Python weekday int)
# Python: Monday=0, Tuesday=1, …, Sunday=6
# ────────────────────────────────────────────────────────────────────────────

_DAY_NAMES: dict[str, int] = {
    # Indonesian
    "senin": 0,
    "selasa": 1,
    "rabu": 2,
    "kamis": 3,
    "jumat": 4,
    "sabtu": 5,
    "minggu": 6,
    # Sundanese
    "senen": 0,
    "salasa": 1,
    "rebo": 2,
    "kemis": 3,
    "jumaah": 4,
    "saptu": 5,
    "ahad": 6,
    # Common STT variants
    "jum'at": 4,
    "jum at": 4,
}

# ────────────────────────────────────────────────────────────────────────────
# Month-name mapping (Indonesian → month int)
# ────────────────────────────────────────────────────────────────────────────

_MONTH_NAMES: dict[str, int] = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "agustus": 8, "agu": 8, "agus": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "desember": 12, "des": 12,
}

# ────────────────────────────────────────────────────────────────────────────
# Relative-date keywords
# ────────────────────────────────────────────────────────────────────────────

_YESTERDAY_WORDS: set[str] = {
    "kemarin", "kemaren", "kamari", "kemari", "kelmarin",
    # Whisper STT variants
    "kemariin", "kemaarin", "kmrn", "kemrn",
}

_DAY_BEFORE_YESTERDAY_PHRASES: set[str] = {
    "kemarin lusa", "kemaren lusa", "lusa kemarin", "lusa kemaren",
    "mangkukna", "kemarin dulu", "kemaren dulu",
}

_TODAY_WORDS: set[str] = {
    "hari ini", "ayeuna", "sekarang",
    # Whisper STT variants
    "hr ini", "harini", "skrg", "skrng", "skg",
    "hari nie", "hariini",
}

_TOMORROW_WORDS: set[str] = {
    "besok", "isukan", "esok",
    # Whisper STT variants (common Whisper errors)
    "isokan", "isok", "bsok", "besook", "bsk",
    "esokan", "esukan",
}

_DAY_AFTER_TOMORROW_PHRASES: set[str] = {
    "lusa", "pageto",
}

# Day prefix words — "hari senin", "poe senen"
_DAY_PREFIX: set[str] = {"hari", "poe"}

# ────────────────────────────────────────────────────────────────────────────
# Pattern compilation
# ────────────────────────────────────────────────────────────────────────────

# Build a regex alternation of all day names (longest first to avoid
# partial matches like "sab" inside "sabtu").
_day_names_sorted = sorted(_DAY_NAMES.keys(), key=len, reverse=True)
_day_names_pattern = "|".join(re.escape(d) for d in _day_names_sorted)

# Build a regex alternation of all month names (longest first)
_month_names_sorted = sorted(_MONTH_NAMES.keys(), key=len, reverse=True)
_month_names_pattern = "|".join(re.escape(m) for m in _month_names_sorted)

# Pattern: [hari|poe] <day_name> [kemarin|lalu|kemari|...]
_WEEKDAY_PAST_RE = re.compile(
    rf"(?:(?:hari|poe)\s+)?({_day_names_pattern})"
    rf"\s+(?:kemarin|kemaren|kamari|kemari|lalu|yang\s+lalu|wingi)",
    re.IGNORECASE,
)

# Pattern: [hari|poe] <day_name> [depan|hareup]
_WEEKDAY_NEXT_RE = re.compile(
    rf"(?:(?:hari|poe)\s+)?({_day_names_pattern})"
    rf"\s+(?:depan|hareup|minggu\s+depan)",
    re.IGNORECASE,
)

# Pattern: "tanggal 4 juni" / "tgl 4 juni" / "4 juni" / "juni 4"
# Optionally with year: "4 juni 2026"
# Year must be 4-digit AND be a plausible year (2000-2099) to avoid
# ambiguity with amount tokens like "25000" being parsed as year "2500 0"
_ABS_DATE_WITH_MONTH_RE = re.compile(
    rf"(?:tanggal\s+|tgl\s+)?(\d{{1,2}})\s+({_month_names_pattern})(?:\s+(20\d{{2}}))?"
    rf"|(?:tanggal\s+|tgl\s+)?({_month_names_pattern})\s+(\d{{1,2}})(?:\s+(20\d{{2}}))?",
    re.IGNORECASE,
)

# Pattern: "tanggal 15" / "tgl 15" — day of current month only
_ABS_DATE_DAY_ONLY_RE = re.compile(
    r"(?:tanggal|tgl)\s+(\d{1,2})\b",
    re.IGNORECASE,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _most_recent_weekday(target_weekday: int, base: date) -> date:
    """Return the most recent date with the given weekday *before* ``base``."""
    days_back = (base.weekday() - target_weekday) % 7
    if days_back == 0:
        days_back = 7
    return base - timedelta(days=days_back)


def _next_weekday(target_weekday: int, base: date) -> date:
    """Return the next date with the given weekday *after* ``base``."""
    days_forward = (target_weekday - base.weekday()) % 7
    if days_forward == 0:
        days_forward = 7
    return base + timedelta(days=days_forward)


def _resolve_absolute_date(day: int, month: int, year: Optional[int], base: date) -> Optional[date]:
    """Resolve an absolute date (day, month, optional year) to a concrete date.

    If year is not provided, picks the nearest occurrence (past or future
    within 6 months). If a year is provided, uses it directly.
    """
    try:
        if year:
            return date(year, month, day)

        # Try current year first
        candidate = date(base.year, month, day)
        # If candidate is more than 6 months in the future, it likely refers to last year
        if (candidate - base).days > 180:
            candidate = date(base.year - 1, month, day)
        # If candidate is more than 6 months in the past, it likely refers to next year
        elif (base - candidate).days > 180:
            candidate = date(base.year + 1, month, day)
        return candidate
    except ValueError:
        return None  # invalid day/month combination


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def parse_date(
    normalized_text: str,
    base_date: Optional[date] = None,
) -> tuple[Optional[str], str]:
    """Parse a relative or absolute date expression from normalized Indonesian text.

    Args:
        normalized_text: Lowercased, normalized transcript text.
        base_date: Reference date (defaults to ``date.today()``).

    Returns:
        A tuple of (date_str, cleaned_text) where:
        - date_str: ISO-8601 date string (``YYYY-MM-DD``) if a date is detected,
          otherwise ``None``.
        - cleaned_text: The input text with the matched date expression removed,
          or the original text if no match was found.
    """
    if not normalized_text:
        return None, normalized_text

    if base_date is None:
        base_date = date.today()

    text = normalized_text.lower().strip()
    original_text = text

    def _validate_and_clean(result_date: date, matched_expr: str) -> tuple[Optional[str], str]:
        """Validate date is within ±365 days and remove matched expression from text."""
        delta = abs((result_date - base_date).days)
        if delta > 365:
            pipeline_logger.warning(
                "[VALIDATION ERROR] | field=date | reason=outside 365-day window | value=%s",
                result_date.isoformat()
            )
            return None, original_text

        # Remove matched expression from text
        cleaned = re.sub(rf"\b{re.escape(matched_expr)}\b", " ", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return result_date.isoformat(), cleaned

    def _clean_span(start: int, end: int) -> str:
        """Remove a character span from text and collapse whitespace."""
        cleaned = text[:start] + " " + text[end:]
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    # ── 1. Day-before-yesterday phrases (check before yesterday) ──────
    for phrase in _DAY_BEFORE_YESTERDAY_PHRASES:
        if phrase in text:
            result_date = base_date - timedelta(days=2)
            return _validate_and_clean(result_date, phrase)

    # ── 2. Specific weekday + past modifier ───────────────────────────
    m = _WEEKDAY_PAST_RE.search(text)
    if m:
        day_name = m.group(1).lower().strip()
        target_weekday = _DAY_NAMES.get(day_name)
        if target_weekday is not None:
            result_date = _most_recent_weekday(target_weekday, base_date)
            matched_expr = m.group(0)
            return _validate_and_clean(result_date, matched_expr)

    # ── 3. Specific weekday + future modifier ─────────────────────────
    m = _WEEKDAY_NEXT_RE.search(text)
    if m:
        day_name = m.group(1).lower().strip()
        target_weekday = _DAY_NAMES.get(day_name)
        if target_weekday is not None:
            result_date = _next_weekday(target_weekday, base_date)
            matched_expr = m.group(0)
            return _validate_and_clean(result_date, matched_expr)

    # ── 4. Yesterday ──────────────────────────────────────────────────
    for word in _YESTERDAY_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            result_date = base_date - timedelta(days=1)
            return _validate_and_clean(result_date, word)

    # ── 5. Today ──────────────────────────────────────────────────────
    for phrase in _TODAY_WORDS:
        if phrase in text:
            return _validate_and_clean(base_date, phrase)

    # ── 6. Tomorrow ───────────────────────────────────────────────────
    for word in _TOMORROW_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            result_date = base_date + timedelta(days=1)
            return _validate_and_clean(result_date, word)

    # ── 7. Day after tomorrow ─────────────────────────────────────────
    for phrase in _DAY_AFTER_TOMORROW_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            result_date = base_date + timedelta(days=2)
            return _validate_and_clean(result_date, phrase)

    # ── 8. Absolute date with month name: "tanggal 4 juni", "4 juni 2026" ──
    m = _ABS_DATE_WITH_MONTH_RE.search(text)
    if m:
        # Two groups depending on which alternative matched
        if m.group(1) and m.group(2):
            # "4 juni" / "tanggal 4 juni"
            day = int(m.group(1))
            month = _MONTH_NAMES.get(m.group(2).lower())
            year = int(m.group(3)) if m.group(3) else None
        else:
            # "juni 4" / "tanggal juni 4"
            month = _MONTH_NAMES.get(m.group(4).lower())
            day = int(m.group(5))
            year = int(m.group(6)) if m.group(6) else None

        if month and 1 <= day <= 31:
            result_date = _resolve_absolute_date(day, month, year, base_date)
            if result_date:
                delta = abs((result_date - base_date).days)
                if delta <= 365:
                    cleaned = _clean_span(m.start(), m.end())
                    return result_date.isoformat(), cleaned
                else:
                    pipeline_logger.warning(
                        "[VALIDATION ERROR] | field=date | reason=outside 365-day window | value=%s",
                        result_date.isoformat()
                    )

    # ── 9. Absolute day-of-month only: "tanggal 15" / "tgl 15" ────────
    m = _ABS_DATE_DAY_ONLY_RE.search(text)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            try:
                result_date = date(base_date.year, base_date.month, day)
                delta = abs((result_date - base_date).days)
                if delta <= 365:
                    cleaned = _clean_span(m.start(), m.end())
                    return result_date.isoformat(), cleaned
            except ValueError:
                pass  # invalid day for current month

    return None, original_text
