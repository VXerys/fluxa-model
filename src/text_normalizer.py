"""Text normalization utilities for Indonesian/Sundanese finance utterances."""

from __future__ import annotations

import re
import unicodedata

_REPLACEMENTS = {
    # ── Amount / currency aliases ─────────────────────────────────────
    "rebu": "ribu",
    "rb": "ribu",
    "ribu rupiah": "ribu",
    "juta rupiah": "juta",

    # ── Sundanese financial verbs ─────────────────────────────────────
    "mayar": "bayar",
    "meser": "beli",
    "meuli": "beli",
    "nampi": "terima",
    "nambut": "pinjam",
    "nginjeum": "pinjam",
    "ngahutang": "hutang",
    "nyimpen": "simpan",
    "nyokot": "ambil",
    "nutup": "tutup",
    
    # ── Common Whisper typos for action verbs ──────────────────────────
    "bli": "beli",
    "byar": "bayar",
    "byyr": "bayar",
    "byr": "bayar",
    "trf": "transfer",
    "trnsfer": "transfer",
    "transferan": "transfer",
    "krm": "kirim",
    "mser": "beli",

    # ── Sundanese prepositions / connectors ────────────────────────────
    "ka ": "ke ",
    "ti ": "dari ",
    "keur ": "untuk ",
    "kanggo ": "untuk ",
    "jeung ": "dan ",
    "sareng ": "dan ",
    "teu ": "tidak ",
    "henteu ": "tidak ",
    
    # ── Common Whisper typos for prepositions ──────────────────────────
    "untk": "untuk",
    "utk": "untuk",
    "bwat": "buat",
    "bwt": "buat",
    "krn": "karena",
    "krna": "karena",
    "gara": "karena",
    "gara gara": "karena",

    # ── Sundanese day names ───────────────────────────────────────────
    "senen": "senin",
    "salasa": "selasa",
    "rebo": "rabu",
    "kemis": "kamis",
    "jumaah": "jumat",
    "saptu": "sabtu",
    "ahad": "minggu",

    # ── Sundanese relative dates ──────────────────────────────────────
    "kamari": "kemarin",
    "kemari": "kemarin",
    "kelmarin": "kemarin",
    "isukan": "besok",
    "isuk": "besok",
    "isokan": "besok",  # Common Whisper typo
    "esukan": "besok",
    "esokan": "besok",
    "pageto": "besok",
    "ayeuna": "hari ini",
    "mangkukna": "kemarin lusa",
    "poe ": "hari ",

    # ── Common Whisper STT typos for time expressions ─────────────────
    "kemaren": "kemarin",
    "kemariin": "kemarin",
    "kemaarin": "kemarin",
    "kmrn": "kemarin",
    "kemrn": "kemarin",
    "harini": "hari ini",
    "hr ini": "hari ini",
    "hari nie": "hari ini",
    "hariini": "hari ini",
    "skrg": "sekarang",
    "skrng": "sekarang",
    "skg": "sekarang",
    "bsok": "besok",
    "besook": "besok",
    "bsk": "besok",

    # ── Sundanese time expressions ────────────────────────────────────
    "isuk-isuk": "pagi",
    "beurang": "siang",
    "sore": "sore",
    "peuting": "malam",
    "wengi": "malam",
    "tadi ": "tadi ",

    # ── Sundanese food / finance domain nouns ──────────────────────────
    "sangu": "nasi",
    "lauk": "ikan",
    "cai": "air",
    "artos": "uang",
    "duit": "uang",
    "gajih": "gaji",
    "ongkos": "ongkos",
    "béaya": "biaya",
    "biaya": "biaya",
    "barang": "barang",

    # ── Sundanese numbers (non-amount context helpers) ─────────────────
    "hiji": "satu",
    "dua": "dua",
    "tilu": "tiga",
    "opat": "empat",
    "genep": "enam",
    "dalapan": "delapan",
    "salapan": "sembilan",

    # ── Common Whisper STT misheard Sundanese ─────────────────────────
    "maser": "beli",
    "maser rasi": "nasi",
    "naon": "apa",
    "teh": "",
    "mah": "",
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
