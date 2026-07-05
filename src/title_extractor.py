"""Rule-based title extractor for Indonesian finance voice transcripts.

Uses verb-anchor extraction to identify transaction titles by scanning for
action verbs and extracting the noun phrase that follows.

Examples:
    "beli nasi padang"                ->  "nasi padang"
    "bayar wifi"                      ->  "wifi"
    "beli kopi kenangan buat lembur"  ->  "kopi kenangan"
"""

from __future__ import annotations

import re


# Transaction action verbs — anchor points for title extraction.
# Including Whisper STT variants and colloquial forms
_ACTION_VERBS: frozenset[str] = frozenset({
    # Standard Indonesian
    "beli", "bayar", "transfer", "kirim", "jajan", 
    "byr", "tf", "trnsfer", "krm", "jajanan",
    # Sundanese
    "mayar", "meser", "meuli", "mser",
    # Colloquial/typo variants
    "bli", "byar", "byyr", "trf", "transferan",
    "beliin", "bayarin", "kirimkan", "jajanan",
    # More action verbs
    "top", "topup", "isi", "ngisi", "tarik", "ambil",
    "bayarin", "beliin", "transfer", "transferin",
    "setor", "nabung", "cairin", "cairkan",
})

# Stop words — boundary markers where title extraction ends.
# Including more variants and STT typos
_STOP_WORDS: frozenset[str] = frozenset({
    # Standard
    "buat", "untuk", "karena", "keur", "kanggo",
    # Variants
    "bwat", "untk", "utk", "krn", "krna",
    "gara", "soalnya", "sebab", "alatan", "margi",
    # Colloquial
    "bwt", "tuk", "wat", "bt",
})

# Noise words to filter out (wallets, connectors, currency, number words, etc.)
# These will be removed until the sequential pipeline is fully implemented.
_NOISE_WORDS: frozenset[str] = frozenset({
    # Prepositions / connectors
    "di", "ke", "ka", "dari", "ti", "pakai", "pake", "lewat", "via", 
    "dengan", "jeung", "sareng", "teu", "henteu", "dan", "tidak",
    "sama", "sm", "dr", "dri",
    # Currency / amount noise
    "rp", "rupiah", "idr", "rupee",
    # Number words (units)
    "nol", "satu", "sa", "hiji", "dua", "tilu", "tiga", "opat",
    "empat", "lima", "genep", "enam", "tujuh", "dalapan", "delapan",
    "salapan", "sembilan",
    # Number words (tens/hundreds/special)
    "sapuluh", "sepuluh", "sabelas", "sebelas",
    "saratus", "seratus", "sarebu", "seribu", "sajuta", "sejuta",
    "puluh", "belas", "ratus",
    # Multipliers
    "ribu", "rebu", "rb", "k", "juta", "jt", "jeti",
    # Wallet names (lowercase)
    "bca", "bri", "bni", "mandiri", "gopay", "go", "pay",
    "dana", "ovo", "shopeepay", "shopee", "cash", "tunai",
    "linkaja", "link", "aja", "jenius", "jago", "blu",
    # Date / day / time words
    "hari", "poe", "ini", "kemarin", "kemaren", "kamari", "kemari",
    "kelmarin", "lusa", "mangkukna", "besok", "isukan", "isuk", "isokan",
    "pageto", "ayeuna", "tadi", "lalu", "depan", "hareup",
    "senin", "senen", "selasa", "salasa", "rabu", "rebo",
    "kamis", "kemis", "jumat", "jumaah", "sabtu", "saptu",
    "minggu", "ahad", "yang",
    # Date prefix words
    "tanggal", "tgl",
    # Month names
    "januari", "jan", "februari", "feb", "maret", "mar",
    "april", "apr", "mei", "juni", "jun", "juli", "jul",
    "agustus", "agu", "agus", "september", "sep", "sept",
    "oktober", "okt", "november", "nov", "desember", "des",
    # Time of day
    "pagi", "siang", "sore", "malam", "peuting", "wengi", "beurang",
    # Common Sundanese particles
    "teh", "mah", "naon", "apa", "oge", "wae", "ge",
    # Common typos/variants
    "skrg", "skrng", "hr", "harini", "bsk",
})


def extract_title(cleaned_text: str) -> str:
    """Extract a clean transaction title from normalized text.

    Args:
        cleaned_text: Text after date and amount tokens have been removed.

    Returns:
        Title string (max 100 chars). Empty string if no valid title found.
    """
    if not cleaned_text:
        return ""

    # Normalize to lowercase and remove punctuation.
    text = cleaned_text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize.
    tokens = text.split()
    if not tokens:
        return ""

    # Step 1: Scan for the first action verb.
    verb_index = -1
    for i, tok in enumerate(tokens):
        if tok in _ACTION_VERBS:
            verb_index = i
            break

    # Step 2: Collect tokens after the verb until a stop word or end.
    if verb_index >= 0:
        # Collect tokens after the verb, skipping noise words.
        candidate_tokens = []
        for i in range(verb_index + 1, len(tokens)):
            if tokens[i] in _STOP_WORDS:
                break
            if tokens[i] not in _NOISE_WORDS:
                candidate_tokens.append(tokens[i])
    else:
        # Fallback: no verb found, collect from start until first stop word, skip noise.
        candidate_tokens = []
        for tok in tokens:
            if tok in _STOP_WORDS:
                break
            if tok not in _NOISE_WORDS:
                candidate_tokens.append(tok)

    # Step 3: Join candidate tokens, also skipping pure-numeric tokens.
    filtered = [t for t in candidate_tokens if not re.fullmatch(r'\d+', t)]
    candidate = " ".join(filtered).strip()

    # Step 4: Validate that the candidate contains at least one alphabetic character.
    if not re.search(r"[a-zA-Z]", candidate):
        return ""

    # Step 5: Truncate to 100 characters at the last complete word boundary.
    if len(candidate) <= 100:
        return candidate

    # Truncate at the last space before the 100-char boundary.
    truncated = candidate[:100]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space]
    return truncated
