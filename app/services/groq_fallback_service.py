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
from datetime import date
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Allowed values (validation)
# ────────────────────────────────────────────────────────────────────────────

CATEGORIES_TAXONOMY: dict[str, list[str]] = {
    "Makan & Minum": [
        "Sarapan", "Makan Siang", "Makan Malam", "Kopi", "Camilan", "Restoran", "Bahan Makanan"
    ],
    "Transportasi": [
        "Bus", "Kereta", "Taksi", "Bensin", "Parkir", "Ojek Online", "Servis Kendaraan"
    ],
    "Belanja": [
        "Pakaian", "Sepatu", "Aksesori", "Elektronik", "Marketplace", "Perawatan Diri"
    ],
    "Rumah": [
        "Sewa", "Listrik", "Air", "Internet", "Furnitur", "Kebersihan", "Perbaikan Rumah"
    ],
    "Hiburan": [
        "Film", "Musik", "Game", "Konser", "Streaming", "Liburan", "Hobi"
    ],
    "Kesehatan": [
        "Dokter", "Obat", "Rumah Sakit", "Asuransi", "Gym", "Vitamin"
    ],
    "Pendidikan": [
        "Kursus", "Buku", "Uang Sekolah", "Seminar", "Sertifikasi", "Alat Tulis"
    ],
    "Tagihan": [
        "Pulsa", "Paket Data", "Langganan", "Kartu Kredit", "Cicilan", "Pajak"
    ],
    "Gaji": [
        "Gaji Bulanan", "Tunjangan", "Lembur", "Bonus Gaji"
    ],
    "Freelance": [
        "Project", "Design", "Development", "Writing", "Konsultasi"
    ],
    "Bisnis": [
        "Penjualan Produk", "Penjualan Jasa", "Keuntungan", "Komisi"
    ],
    "Investasi": [
        "Dividen", "Bunga", "Capital Gain", "Crypto", "Saham"
    ],
    "Hadiah": [
        "Keluarga", "Teman", "Reward", "Hadiah Lomba"
    ],
    "Pengembalian": [
        "Cashback", "Refund", "Reimbursement", "Utang Dibayar"
    ],
    "Lainnya": [
        "Hadiah Keluar", "Donasi", "Biaya Admin", "Tak Terduga", "Pengeluaran Lainnya", "Pemasukan Lainnya"
    ],
    "Transfer": []
}

ALLOWED_CATEGORIES: set[str] = set(CATEGORIES_TAXONOMY.keys())

ALLOWED_TYPES: set[str] = {"expense", "income", "transfer"}

ALLOWED_WALLETS: dict[str, str] = {
    # "bca": "BCA",
    # "bri": "BRI",
    # "bni": "BNI",
    # "mandiri": "Mandiri",
    # "dana": "DANA",
    # "gopay": "GoPay",
    # "go pay": "GoPay",
    # "ovo": "OVO",
    # "shopeepay": "ShopeePay",
    # "shopee pay": "ShopeePay",
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
    "cash", "tunai",
}

_TRANSFER_PHRASES: set[str] = {
    "transfer", "kirim", "pindah", "pindahin", "tf",
}

_SUSPICIOUS_STT_WORDS: set[str] = {
    "hmm", "eh", "uh", "uhm", "hah", "kok", "loh",
    "aduh", "duh", "yah",
}

# Sundanese-dialect words that signal the transcript needs LLM correction.
_SUNDANESE_KEYWORDS: set[str] = {
    "meser", "meuli", "mayar", "nampi", "nambut", "nginjeum",
    "nyimpen", "nyokot", "nutup", "ngahutang",
    "kamari", "kemari", "kelmarin", "isukan", "isuk", "pageto",
    "ayeuna", "mangkukna", "poe",
    "senen", "salasa", "rebo", "kemis", "jumaah", "saptu", "ahad",
    "artos", "duit", "sangu", "lauk", "cai",
    "ka", "ti", "keur", "kanggo", "jeung", "sareng",
    "teh", "mah", "naon",
    "maser",
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
# Sundanese detection helper
# ────────────────────────────────────────────────────────────────────────────

def _transcript_contains_sundanese(text: str) -> bool:
    """Return True if the transcript contains Sundanese-dialect words."""
    tokens = set(text.lower().split())
    return bool(tokens & _SUNDANESE_KEYWORDS)


# ────────────────────────────────────────────────────────────────────────────
# Groq prompt
# ────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
Kamu adalah asisten koreksi transaksi keuangan Indonesia yang sangat paham \
bahasa Sunda (Sundanese).

Tanggal hari ini: {today_date}

Kamu akan menerima hasil parsing transaksi dari speech-to-text (Whisper) \
yang mungkin mengandung kesalahan pada judul, deskripsi, kategori, wallet, \
tipe transaksi, atau tanggal. Transkrip sering menggunakan bahasa Sunda \
atau campuran Sunda-Indonesia.

### Pemahaman Bahasa Sunda
Kamu HARUS memahami kosakata Sunda berikut dan mengonversinya ke \
Bahasa Indonesia yang bersih:

**Kata kerja keuangan:**
- meser / meuli = beli
- mayar = bayar
- nampi = terima
- nambut / nginjeum = pinjam
- ngahutang = hutang
- nyimpen = simpan
- nyokot = ambil

**Makanan & benda:**
- sangu = nasi
- lauk = ikan/lauk-pauk
- cai = air/minuman
- artos / duit = uang

**Waktu & tanggal (Sunda):**
- kamari / kemari = kemarin
- isukan / isuk = besok
- ayeuna = hari ini
- mangkukna = kemarin lusa (2 hari lalu)
- poe = hari
- senen = senin, salasa = selasa, rebo = rabu
- kemis = kamis, jumaah = jumat, saptu = sabtu, ahad = minggu

**Preposisi:**
- ka = ke, ti = dari, keur/kanggo = untuk
- jeung/sareng = dan/dengan

Contoh konversi:
- "meser sangu padang kamari" → title: "Nasi Padang", description: "Beli nasi padang kemarin"
- "mayar ongkos angkot poe senen kemari" → title: "Ongkos Angkot", description: "Bayar ongkos angkot hari Senin kemarin"
- "nampi gajih" → title: "Gaji", description: "Terima gaji"
- "meuli obat ti apotek" → title: "Obat", description: "Beli obat dari apotek"

### Kategori & Subkategori (Sangat Penting):
Kamu harus memilih kategori utama (category) dan subkategori (subcategory) yang paling sesuai dari daftar di bawah ini. Jangan membuat kategori atau subkategori di luar daftar ini!

Daftar Kategori Utama & Subkategori yang Diizinkan:
1. Makan & Minum
   Subkategori: Sarapan, Makan Siang, Makan Malam, Kopi, Camilan, Restoran, Bahan Makanan
2. Transportasi
   Subkategori: Bus, Kereta, Taksi, Bensin, Parkir, Ojek Online, Servis Kendaraan
3. Belanja
   Subkategori: Pakaian, Sepatu, Aksesori, Elektronik, Marketplace, Perawatan Diri
4. Rumah
   Subkategori: Sewa, Listrik, Air, Internet, Furnitur, Kebersihan, Perbaikan Rumah
5. Hiburan
   Subkategori: Film, Musik, Game, Concert (Konser), Streaming, Liburan, Hobi
6. Kesehatan
   Subkategori: Dokter, Obat, Rumah Sakit, Asuransi, Gym, Vitamin
7. Pendidikan
   Subkategori: Kursus, Buku, Uang Sekolah, Seminar, Sertifikasi, Alat Tulis
8. Tagihan
   Subkategori: Pulsa, Paket Data, Langganan, Kartu Kredit, Cicilan, Pajak
9. Gaji
   Subkategori: Gaji Bulanan, Tunjangan, Lembur, Bonus Gaji
10. Freelance
    Subkategori: Project, Design, Development, Writing, Konsultasi
11. Bisnis
    Subkategori: Penjualan Produk, Penjualan Jasa, Keuntungan, Komisi
12. Investasi
    Subkategori: Dividen, Bunga, Capital Gain, Crypto, Saham
13. Hadiah
    Subkategori: Keluarga, Teman, Reward, Hadiah Lomba
14. Pengembalian
    Subkategori: Cashback, Refund, Reimbursement, Utang Dibayar
15. Lainnya
    Subkategori: Hadiah Keluar, Donasi, Biaya Admin, Tak Terduga, Pengeluaran Lainnya, Pemasukan Lainnya
16. Transfer
    Subkategori: (Tidak ada subkategori, gunakan null)

### Tugasmu:
1. Perbaiki **title** agar berisi nama item/tujuan transaksi yang bersih \
   dalam Bahasa Indonesia, tanpa angka nominal, tanpa nama wallet, tanpa \
   kata tanggal/waktu. Gunakan huruf kapital di awal setiap kata.
2. Buat **description** yang merupakan kalimat pendek (1 kalimat) dalam \
   Bahasa Indonesia yang menjelaskan transaksi secara natural. \
   SELALU berikan description yang bermakna, jangan pernah kirim null \
   untuk description.
3. Perbaiki **category** jika tidak sesuai konteks. Gunakan salah satu dari 16 kategori utama di atas.
4. Tentukan **subcategory** yang paling sesuai dari subkategori yang diperbolehkan untuk kategori utama tersebut. Jika tidak ada subkategori yang cocok, gunakan null.
5. Perbaiki **type** jika tidak konsisten. Tipe yang diizinkan: \
   expense, income, transfer.
6. Perbaiki **wallet** jika terdeteksi di transkrip. Wallet yang diizinkan: \
   Cash.
7. Perbaiki **date** jika ada ekspresi tanggal relatif. Gunakan format \
   YYYY-MM-DD. Referensi: hari ini = {today_date}.

Aturan penting:
- Jangan mengubah nominal/amount.
- Jika suatu field sudah benar, kirim null untuk field tersebut.
- Untuk **description**, SELALU berikan kalimat yang bermakna.
- Jika tidak yakin, kirim null (kecuali description).
- Jawab HANYA dengan JSON valid, tanpa teks lain.

Format respons (JSON saja):
{{
  "title": "string atau null",
  "description": "string (WAJIB, jangan null)",
  "type": "expense|income|transfer atau null",
  "category": "kategori utama yang diizinkan atau null",
  "subcategory": "subkategori yang diizinkan untuk kategori tersebut atau null",
  "wallet": "wallet yang diizinkan atau null",
  "date": "YYYY-MM-DD atau null",
  "reason": "alasan singkat koreksi"
}}"""

def _build_system_prompt() -> str:
    """Build the system prompt with today's date injected."""
    return _SYSTEM_PROMPT_TEMPLATE.format(today_date=date.today().isoformat())


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
            "local_date": transaction.get("date"),
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
    if value == "Tagihan & Utilitas":
        return "Tagihan"
    return value if value in ALLOWED_CATEGORIES else None


def _validate_subcategory(category: str | None, subcategory: str | None) -> str | None:
    if not category or not subcategory:
        return None
    allowed_subs = CATEGORIES_TAXONOMY.get(category, [])
    sub_clean = subcategory.strip().lower()
    for allowed in allowed_subs:
        if allowed.lower() == sub_clean:
            return allowed
    return None


def _validate_type(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower().strip()
    return lowered if lowered in ALLOWED_TYPES else None


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str | None) -> bool:
    """Return True if value is a valid YYYY-MM-DD date string."""
    if value is None:
        return False
    if not _DATE_RE.match(value.strip()):
        return False
    try:
        parts = value.strip().split("-")
        date(int(parts[0]), int(parts[1]), int(parts[2]))
        return True
    except (ValueError, IndexError):
        return False


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

        # 7. Sundanese dialect detected — needs LLM for clean title/description
        if _transcript_contains_sundanese(raw_text):
            return True

        # 8. Description is empty or same as title — needs enrichment
        local_desc = transaction.get("description")
        if not local_desc or local_desc == local_title:
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
                {"role": "system", "content": _build_system_prompt()},
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

        # Subcategory
        final_category = groq_category or tx.get("category")
        groq_subcategory = _validate_subcategory(final_category, groq_response.get("subcategory"))
        if groq_subcategory and groq_subcategory != tx.get("subcategory"):
            tx["subcategory"] = groq_subcategory
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

        # Date
        groq_date = groq_response.get("date")
        if groq_date and isinstance(groq_date, str) and _validate_date(groq_date):
            if groq_date != tx.get("date"):
                tx["date"] = groq_date
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
