"""Text normalization utilities for Indonesian/Sundanese finance utterances."""

from __future__ import annotations

import re
import unicodedata

_REPLACEMENTS = {
    "rebu": "ribu",
    "rb": "ribu",
    "ribu rupiah": "ribu",
    "juta rupiah": "juta",
    "mayar": "bayar",
    "meser": "beli",
    "meuli": "beli",
    "nampi": "terima",
    "ka ": "ke ",
    "ti ": "dari ",
}


def normalize_text(text: str) -> str:
    """Normalize text without destroying natural Sunda/Indo variation.

    This is intentionally conservative: it lowercases, strips punctuation that is
    usually noise, normalizes spacing, and maps frequent finance-domain variants.
    """
    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", str(text)).lower().strip()
    value = re.sub(r"[\t\n\r]+", " ", value)
    value = re.sub(r"[,.!?;:]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    # Word-boundary replacements for short tokens; phrase replacements for others.
    for src, dst in _REPLACEMENTS.items():
        if src.endswith(" ") or " " in src:
            value = value.replace(src, dst)
        else:
            value = re.sub(rf"\b{re.escape(src)}\b", dst, value)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_wallet_name(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = normalize_text(value)
    aliases = {
        "bca": "BCA",
        "bri": "BRI",
        "bni": "BNI",
        "mandiri": "Mandiri",
        "gopay": "GoPay",
        "go pay": "GoPay",
        "dana": "DANA",
        "ovo": "OVO",
        "shopeepay": "ShopeePay",
        "shopee pay": "ShopeePay",
        "cash": "Cash",
        "tunai": "Cash",
    }
    return aliases.get(normalized, value)
