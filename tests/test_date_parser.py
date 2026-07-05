"""Unit tests for src.date_parser.

Tests cover all Sundanese/Indonesian relative date expressions:
- kemarin / kamari / kemari / kelmarin
- kemarin lusa / mangkukna
- hari ini / ayeuna
- besok / isukan / esok
- hari <day_name> kemarin (Indonesian)
- poe <day_name> kemari (Sundanese)
- <day_name> depan (future)
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.date_parser import parse_date


# ────────────────────────────────────────────────────────────────────────────
# Reference date: Wednesday, 2026-06-17
# ────────────────────────────────────────────────────────────────────────────

BASE = date(2026, 6, 17)  # Wednesday


class TestYesterday:
    """Tests for 'kemarin', 'kamari', 'kemari', 'kelmarin'."""

    def test_kemarin(self) -> None:
        date_str, cleaned = parse_date("beli nasi kemarin", BASE)
        assert date_str == "2026-06-16"
        assert cleaned == "beli nasi"

    def test_kamari(self) -> None:
        date_str, cleaned = parse_date("meser sangu kamari", BASE)
        assert date_str == "2026-06-16"
        assert cleaned == "meser sangu"

    def test_kemari(self) -> None:
        date_str, cleaned = parse_date("bayar ongkos kemari", BASE)
        assert date_str == "2026-06-16"
        assert cleaned == "bayar ongkos"

    def test_kelmarin(self) -> None:
        date_str, cleaned = parse_date("beli obat kelmarin", BASE)
        assert date_str == "2026-06-16"
        assert cleaned == "beli obat"

    def test_kemarin_embedded(self) -> None:
        date_str, cleaned = parse_date("beli seblak 15 ribu kemarin", BASE)
        assert date_str == "2026-06-16"
        assert cleaned == "beli seblak 15 ribu"


class TestDayBeforeYesterday:
    """Tests for 'kemarin lusa', 'mangkukna'."""

    def test_kemarin_lusa(self) -> None:
        date_str, cleaned = parse_date("bayar wifi kemarin lusa", BASE)
        assert date_str == "2026-06-15"
        assert cleaned == "bayar wifi"

    def test_mangkukna(self) -> None:
        date_str, cleaned = parse_date("meser cai mangkukna", BASE)
        assert date_str == "2026-06-15"
        assert cleaned == "meser cai"

    def test_lusa_kemarin(self) -> None:
        date_str, cleaned = parse_date("beli nasi lusa kemarin", BASE)
        assert date_str == "2026-06-15"
        assert cleaned == "beli nasi"


class TestToday:
    """Tests for 'hari ini', 'ayeuna'."""

    def test_hari_ini(self) -> None:
        date_str, cleaned = parse_date("beli nasi hari ini", BASE)
        assert date_str == "2026-06-17"
        assert cleaned == "beli nasi"

    def test_ayeuna(self) -> None:
        date_str, cleaned = parse_date("mayar artos ayeuna", BASE)
        assert date_str == "2026-06-17"
        assert cleaned == "mayar artos"

    def test_sekarang(self) -> None:
        """Test new 'sekarang' keyword for today."""
        date_str, cleaned = parse_date("beli kopi sekarang", BASE)
        assert date_str == "2026-06-17"
        assert cleaned == "beli kopi"


class TestTomorrow:
    """Tests for 'besok', 'isukan', 'esok'."""

    def test_besok(self) -> None:
        date_str, cleaned = parse_date("bayar tagihan besok", BASE)
        assert date_str == "2026-06-18"
        assert cleaned == "bayar tagihan"

    def test_isukan(self) -> None:
        date_str, cleaned = parse_date("meser obat isukan", BASE)
        assert date_str == "2026-06-18"
        assert cleaned == "meser obat"

    def test_esok(self) -> None:
        date_str, cleaned = parse_date("kirim uang esok", BASE)
        assert date_str == "2026-06-18"
        assert cleaned == "kirim uang"


class TestDayAfterTomorrow:
    """Tests for 'lusa', 'pageto'."""

    def test_lusa(self) -> None:
        date_str, cleaned = parse_date("bayar tagihan lusa", BASE)
        assert date_str == "2026-06-19"
        assert cleaned == "bayar tagihan"

    def test_pageto(self) -> None:
        date_str, cleaned = parse_date("meser obat pageto", BASE)
        assert date_str == "2026-06-19"
        assert cleaned == "meser obat"


class TestWeekdayPastIndonesian:
    """Tests for 'hari <day> kemarin' / '<day> kemarin' (Indonesian)."""

    def test_senin_kemarin(self) -> None:
        # BASE is Wed 2026-06-17, most recent Monday = 2026-06-15
        date_str, _ = parse_date("hari senin kemarin", BASE)
        assert date_str == "2026-06-15"

    def test_selasa_kemarin(self) -> None:
        # Most recent Tuesday = 2026-06-16
        date_str, _ = parse_date("selasa kemarin", BASE)
        assert date_str == "2026-06-16"

    def test_jumat_kemarin(self) -> None:
        # Most recent Friday before Wed = 2026-06-12
        date_str, _ = parse_date("jumat kemarin", BASE)
        assert date_str == "2026-06-12"

    def test_rabu_kemarin(self) -> None:
        # BASE is Wed, so "rabu kemarin" = previous Wed = 2026-06-10
        date_str, _ = parse_date("hari rabu kemarin", BASE)
        assert date_str == "2026-06-10"

    def test_sabtu_lalu(self) -> None:
        # Most recent Saturday before Wed = 2026-06-13
        date_str, _ = parse_date("sabtu lalu", BASE)
        assert date_str == "2026-06-13"

    def test_minggu_yang_lalu(self) -> None:
        # Most recent Sunday before Wed = 2026-06-14
        date_str, _ = parse_date("minggu yang lalu", BASE)
        assert date_str == "2026-06-14"


class TestWeekdayPastSundanese:
    """Tests for 'poe <day_name> kemari' (Sundanese)."""

    def test_poe_senen_kemari(self) -> None:
        # BASE = Wed, most recent Monday = 2026-06-15
        date_str, _ = parse_date("poe senen kemari", BASE)
        assert date_str == "2026-06-15"

    def test_poe_salasa_kemari(self) -> None:
        # Most recent Tuesday = 2026-06-16
        date_str, _ = parse_date("poe salasa kemari", BASE)
        assert date_str == "2026-06-16"

    def test_poe_rebo_kemari(self) -> None:
        # Most recent Wednesday (not today) = 2026-06-10
        date_str, _ = parse_date("poe rebo kemari", BASE)
        assert date_str == "2026-06-10"

    def test_poe_kemis_kemari(self) -> None:
        # Most recent Thursday = 2026-06-11
        date_str, _ = parse_date("poe kemis kemari", BASE)
        assert date_str == "2026-06-11"

    def test_poe_jumaah_kemari(self) -> None:
        # Most recent Friday = 2026-06-12
        date_str, _ = parse_date("poe jumaah kemari", BASE)
        assert date_str == "2026-06-12"

    def test_poe_saptu_kemari(self) -> None:
        # Most recent Saturday = 2026-06-13
        date_str, _ = parse_date("poe saptu kemari", BASE)
        assert date_str == "2026-06-13"

    def test_poe_ahad_kemari(self) -> None:
        # Most recent Sunday = 2026-06-14
        date_str, _ = parse_date("poe ahad kemari", BASE)
        assert date_str == "2026-06-14"


class TestWeekdayFuture:
    """Tests for '<day> depan'."""

    def test_senin_depan(self) -> None:
        # BASE = Wed 2026-06-17, next Monday = 2026-06-22
        date_str, _ = parse_date("senin depan", BASE)
        assert date_str == "2026-06-22"

    def test_jumat_depan(self) -> None:
        # Next Friday = 2026-06-19
        date_str, _ = parse_date("jumat depan", BASE)
        assert date_str == "2026-06-19"


class TestNoDateDetected:
    """Tests that non-date text returns (None, original_text)."""

    def test_plain_transaction(self) -> None:
        date_str, cleaned = parse_date("beli nasi padang 21 ribu pakai bca", BASE)
        assert date_str is None
        assert cleaned == "beli nasi padang 21 ribu pakai bca"

    def test_empty_string(self) -> None:
        date_str, cleaned = parse_date("", BASE)
        assert date_str is None

    def test_amount_only(self) -> None:
        date_str, cleaned = parse_date("dua puluh satu ribu", BASE)
        assert date_str is None
        assert cleaned == "dua puluh satu ribu"


class TestEmbeddedInSentence:
    """Tests that date expressions are detected embedded in longer sentences."""

    def test_beli_seblak_kamari(self) -> None:
        date_str, _ = parse_date("meser seblak 15 ribu kamari", BASE)
        assert date_str == "2026-06-16"

    def test_angkot_poe_senen_kemari(self) -> None:
        date_str, _ = parse_date("ongkos angkot 5 ribu poe senen kemari", BASE)
        assert date_str == "2026-06-15"

    def test_bayar_wifi_kemarin_lusa(self) -> None:
        date_str, _ = parse_date("bayar wifi rp300000 kemarin lusa", BASE)
        assert date_str == "2026-06-15"

    def test_gaji_hari_jumat_kemarin(self) -> None:
        date_str, _ = parse_date("terima gaji hari jumat kemarin", BASE)
        assert date_str == "2026-06-12"

    def test_besok_embedded(self) -> None:
        date_str, _ = parse_date("bayar tagihan listrik besok", BASE)
        assert date_str == "2026-06-18"


class TestDefaultBaseDate:
    """Test that parse_date uses today when no base_date is given."""

    def test_today_default(self) -> None:
        today = date.today()
        date_str, _ = parse_date("beli nasi kemarin")
        expected = (today - timedelta(days=1)).isoformat()
        assert date_str == expected
