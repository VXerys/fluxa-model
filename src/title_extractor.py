"""Rule-based title extractor for Indonesian finance voice transcripts.

Strips amount tokens, wallet names, currency words, and common transaction
verbs from the normalized text to produce a clean title candidate.

Examples:
    "beli nasi padang rp21 ribu pakai bca"  ->  "nasi padang"
    "12 ribu jeruk nipis dengan bca"        ->  "jeruk nipis"
    "transfer bca ke gopay 50 ribu"         ->  ""
"""

from __future__ import annotations

import re


# Words to strip — they are not part of the transaction "item" title.
_STRIP_WORDS: set[str] = {
    # Transaction verbs
    "beli", "buat", "bayar", "mayar", "meser", "meuli", "jajan",
    "terima", "nampi", "dapat", "dapet", "kirim", "transfer",
    "pindah", "setor", "tarik", "nambut", "nginjeum", "ngahutang",
    "nyimpen", "nyokot", "nutup", "simpan", "ambil", "tutup",
    "pinjam", "hutang",
    # Prepositions / connectors
    "di", "ke", "ka", "dari", "ti", "untuk", "buat", "pakai",
    "pake", "lewat", "via", "dengan", "keur", "kanggo", "jeung",
    "sareng", "teu", "henteu", "dan", "tidak",
    # Currency / amount noise
    "rp", "rupiah", "idr",
    # Number words (units)
    "nol", "satu", "sa", "hiji", "dua", "tilu", "tiga", "opat",
    "empat", "lima", "genep", "enam", "tujuh", "dalapan", "delapan",
    "salapan", "sembilan",
    # Number words (tens/hundreds/special)
    "sapuluh", "sepuluh", "sabelas", "sebelas",
    "saratus", "seratus", "sarebu", "seribu", "sajuta", "sejuta",
    "puluh", "belas", "ratus",
    # Multipliers
    "ribu", "rebu", "rb", "k", "juta", "jt",
    # Wallet names (lowercase)
    "bca", "bri", "bni", "mandiri", "gopay", "go", "pay",
    "dana", "ovo", "shopeepay", "shopee", "cash", "tunai",
    # Date / day / time words (should not be part of title)
    "hari", "poe", "ini", "kemarin", "kemaren", "kamari", "kemari",
    "kelmarin", "lusa", "mangkukna", "besok", "isukan", "isuk",
    "pageto", "ayeuna", "tadi", "lalu", "depan", "hareup",
    "senin", "senen", "selasa", "salasa", "rabu", "rebo",
    "kamis", "kemis", "jumat", "jumaah", "sabtu", "saptu",
    "minggu", "ahad", "yang",
    # Time of day
    "pagi", "siang", "sore", "malam", "peuting", "wengi",
    "beurang",
    # Common Sundanese particles (no meaning in title)
    "teh", "mah", "naon", "apa",
}


# Regex to strip standalone digits and "rpNN" patterns.
_DIGIT_PATTERN = re.compile(r"\b(?:rp\s*)?\d+\b", re.IGNORECASE)


def extract_title(normalized_text: str) -> str:
    """Extract a clean transaction title from normalized text.

    Args:
        normalized_text: Text after ``normalize_text()`` from the text normalizer.

    Returns:
        Cleaned title string.  May be empty if no meaningful words remain.
    """
    if not normalized_text:
        return ""

    # Remove digit/amount patterns first.
    cleaned = _DIGIT_PATTERN.sub(" ", normalized_text.lower())

    # Remove punctuation leftovers.
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)

    tokens = cleaned.split()

    # Keep only tokens that are not in the strip set.
    kept = [tok for tok in tokens if tok not in _STRIP_WORDS and len(tok) > 1]

    return " ".join(kept).strip()
