"""Rule-based relative date parser for Indonesian/Sundanese voice transcripts.

Resolves relative time expressions that commonly appear in voice-based
financial transactions:

    - "kemarin" / "kamari"          → yesterday
    - "kemarin lusa" / "mangkukna"  → day before yesterday
    - "hari ini" / "ayeuna"         → today
    - "besok" / "isukan"            → tomorrow
    - "senin kemarin"               → most recent past Monday
    - "poe senen kemari"            → most recent past Monday (Sundanese)
    - "hari jumat kemarin"          → most recent past Friday

All outputs are ISO-8601 date strings (``YYYY-MM-DD``).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

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
    "jumat": 4,
    "jum at": 4,
}

# ────────────────────────────────────────────────────────────────────────────
# Relative-date keywords
# ────────────────────────────────────────────────────────────────────────────

_YESTERDAY_WORDS: set[str] = {
    "kemarin", "kemaren", "kamari", "kemari", "kelmarin",
}

_DAY_BEFORE_YESTERDAY_PHRASES: set[str] = {
    "kemarin lusa", "kemaren lusa", "lusa kemarin", "lusa kemaren",
    "mangkukna",
}

_TODAY_WORDS: set[str] = {
    "hari ini", "ayeuna",
}

_TOMORROW_WORDS: set[str] = {
    "besok", "isukan", "esok",
}

_DAY_AFTER_TOMORROW_PHRASES: set[str] = {
    "lusa", "pageto",
}

_PAST_MODIFIER_WORDS: set[str] = {
    "kemarin", "kemaren", "kamari", "kemari", "lalu", "lalu",
    "yang lalu", "wingi", "tadi",
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

# Pattern: [hari|poe] <day_name> [kemarin|lalu|kemari|...]
# e.g. "hari senin kemarin", "poe senen kemari", "selasa lalu", "jumat kemarin"
_WEEKDAY_PAST_RE = re.compile(
    rf"(?:(?:hari|poe)\s+)?({_day_names_pattern})"
    rf"\s+(?:kemarin|kemaren|kamari|kemari|lalu|yang\s+lalu|wingi)",
    re.IGNORECASE,
)

# Pattern: [hari|poe] <day_name> [depan|hareup]
# e.g. "hari senin depan", "jumat depan"
_WEEKDAY_NEXT_RE = re.compile(
    rf"(?:(?:hari|poe)\s+)?({_day_names_pattern})"
    rf"\s+(?:depan|hareup|minggu\s+depan)",
    re.IGNORECASE,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _most_recent_weekday(target_weekday: int, base: date) -> date:
    """Return the most recent date with the given weekday *before* ``base``.

    If ``base`` itself is the target weekday, we go back 7 days (i.e. "last
    Monday" when today is Monday means the previous Monday).
    """
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


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def parse_date(
    normalized_text: str,
    base_date: Optional[date] = None,
) -> Optional[str]:
    """Parse a relative date expression from normalized Indonesian/Sundanese text.

    Args:
        normalized_text: Lowercased, normalized transcript text.
        base_date: Reference date (defaults to ``date.today()``).

    Returns:
        ISO-8601 date string (``YYYY-MM-DD``) if a relative date is
        detected, otherwise ``None``.
    """
    if not normalized_text:
        return None

    if base_date is None:
        base_date = date.today()

    text = normalized_text.lower().strip()

    # ── 1. Day-before-yesterday phrases (check before yesterday) ──────
    for phrase in _DAY_BEFORE_YESTERDAY_PHRASES:
        if phrase in text:
            return (base_date - timedelta(days=2)).isoformat()

    # ── 2. Specific weekday + past modifier ───────────────────────────
    m = _WEEKDAY_PAST_RE.search(text)
    if m:
        day_name = m.group(1).lower().strip()
        target_weekday = _DAY_NAMES.get(day_name)
        if target_weekday is not None:
            return _most_recent_weekday(target_weekday, base_date).isoformat()

    # ── 3. Specific weekday + future modifier ─────────────────────────
    m = _WEEKDAY_NEXT_RE.search(text)
    if m:
        day_name = m.group(1).lower().strip()
        target_weekday = _DAY_NAMES.get(day_name)
        if target_weekday is not None:
            return _next_weekday(target_weekday, base_date).isoformat()

    # ── 4. Yesterday ──────────────────────────────────────────────────
    for word in _YESTERDAY_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return (base_date - timedelta(days=1)).isoformat()

    # ── 5. Today ──────────────────────────────────────────────────────
    for phrase in _TODAY_WORDS:
        if phrase in text:
            return base_date.isoformat()

    # ── 6. Tomorrow ───────────────────────────────────────────────────
    for word in _TOMORROW_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return (base_date + timedelta(days=1)).isoformat()

    # ── 7. Day after tomorrow ─────────────────────────────────────────
    for phrase in _DAY_AFTER_TOMORROW_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return (base_date + timedelta(days=2)).isoformat()

    return None
