"""Unit tests for src.title_extractor."""

from __future__ import annotations

from src.title_extractor import extract_title


class TestExtractTitle:
    """Tests for the rule-based title extractor."""

    def test_nasi_padang_with_amount_and_wallet(self) -> None:
        # "beli nasi padang rp21 ribu pakai bca" -> normalized by text_normalizer
        normalized = "beli nasi padang rp21 ribu pakai bca"
        result = extract_title(normalized)
        assert "nasi" in result.lower()
        assert "padang" in result.lower()
        # Should NOT contain amount or wallet noise
        assert "rp" not in result.lower()
        assert "21" not in result
        assert "bca" not in result.lower()

    def test_jeruk_nipis(self) -> None:
        normalized = "12 ribu jeruk nipis dengan bca"
        result = extract_title(normalized)
        assert "jeruk" in result.lower()
        assert "nipis" in result.lower()
        assert "bca" not in result.lower()
        assert "12" not in result

    def test_transfer_returns_empty(self) -> None:
        normalized = "transfer bca ke gopay 50 ribu"
        result = extract_title(normalized)
        # Transfer has no item title — expect empty or very minimal
        assert "bca" not in result.lower()
        assert "gopay" not in result.lower()
        assert "50" not in result

    def test_kopi_susu(self) -> None:
        normalized = "beli kopi susu 15 ribu pakai dana"
        result = extract_title(normalized)
        assert "kopi" in result.lower()
        assert "susu" in result.lower()
        assert "dana" not in result.lower()

    def test_empty_input(self) -> None:
        assert extract_title("") == ""

    def test_only_amount(self) -> None:
        # If the transcript is ONLY amount words, should return empty
        normalized = "dua puluh satu ribu"
        result = extract_title(normalized)
        assert result.strip() == ""

    def test_bayar_wifi(self) -> None:
        normalized = "bayar wifi rp 300000"
        result = extract_title(normalized)
        assert "wifi" in result.lower()
        assert "300000" not in result

    def test_obat_apotek(self) -> None:
        normalized = "beli obat di apotek 25 ribu cash"
        result = extract_title(normalized)
        assert "obat" in result.lower()
        assert "apotek" in result.lower()
        assert "cash" not in result.lower()
