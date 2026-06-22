"""Unit tests for app.services.groq_fallback_service.

All Groq API calls are mocked — no real API key needed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from app.services.groq_fallback_service import (
    GroqFallbackService,
    _title_looks_bad,
    _transcript_contains_wallet,
    _transcript_contains_transfer,
    _transcript_looks_noisy,
    _validate_category,
    _validate_type,
    _validate_wallet,
    ALLOWED_CATEGORIES,
    ALLOWED_TYPES,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_local_result(
    raw: str = "beli nasi padang rp21 ribu pakai tunai",
    normalized: str = "beli nasi padang rp21 ribu pakai tunai",
    amount: int | None = 21000,
    tx_type: str = "expense",
    category: str = "Makan & Minum",
    wallet: str | None = None,
    title: str | None = None,
    description: str | None = None,
    warnings: list[str] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "transcript": {
            "raw": raw,
            "normalized": normalized,
            "language_hint": "id",
            "confidence": confidence,
        },
        "transaction": {
            "type": tx_type,
            "amount": amount,
            "category": category,
            "wallet": wallet,
            "title": title,
            "description": description,
            "currency": "IDR",
        },
        "classification": {
            "raw_type": tx_type,
            "resolved_type": tx_type,
            "category": category,
        },
        "warnings": warnings if warnings is not None else [],
    }


def _make_enabled_service() -> GroqFallbackService:
    """Create a service instance with feature flag enabled and a fake key."""
    svc = GroqFallbackService()
    svc._enabled = True
    svc._api_key = "fake-test-key"
    return svc


# ────────────────────────────────────────────────────────────────────────────
# Tests: Validation helpers
# ────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_validate_category_valid(self) -> None:
        for cat in ALLOWED_CATEGORIES:
            assert _validate_category(cat) == cat

    def test_validate_category_invalid(self) -> None:
        assert _validate_category("InvalidCategory") is None
        assert _validate_category("") is None

    def test_validate_category_none(self) -> None:
        assert _validate_category(None) is None

    def test_validate_type_valid(self) -> None:
        for t in ALLOWED_TYPES:
            assert _validate_type(t) == t

    def test_validate_type_case_insensitive(self) -> None:
        assert _validate_type("EXPENSE") == "expense"
        assert _validate_type("Income") == "income"

    def test_validate_type_invalid(self) -> None:
        assert _validate_type("refund") is None

    def test_validate_wallet_canonical(self) -> None:
        assert _validate_wallet("Cash") == "Cash"
        assert _validate_wallet("BCA") is None

    def test_validate_wallet_alias(self) -> None:
        assert _validate_wallet("gopay") is None
        assert _validate_wallet("tunai") == "Cash"

    def test_validate_wallet_invalid(self) -> None:
        assert _validate_wallet("Tokopedia") is None

    def test_validate_wallet_none(self) -> None:
        assert _validate_wallet(None) is None


# ────────────────────────────────────────────────────────────────────────────
# Tests: Trigger condition helpers
# ────────────────────────────────────────────────────────────────────────────

class TestTriggerHelpers:
    def test_title_looks_bad_empty(self) -> None:
        assert _title_looks_bad(None) is True
        assert _title_looks_bad("") is True
        assert _title_looks_bad("ab") is True  # too short

    def test_title_looks_bad_with_amount_noise(self) -> None:
        assert _title_looks_bad("Maser rasi padang rp21") is True
        assert _title_looks_bad("beli 5000 nasi") is True

    def test_title_looks_good(self) -> None:
        assert _title_looks_bad("Nasi padang") is False

    def test_transcript_contains_wallet(self) -> None:
        assert _transcript_contains_wallet("beli pakai tunai") is True
        assert _transcript_contains_wallet("pakai cash") is True
        assert _transcript_contains_wallet("beli nasi padang") is False

    def test_transcript_contains_transfer(self) -> None:
        assert _transcript_contains_transfer("transfer bca ke gopay") is True
        assert _transcript_contains_transfer("kirim uang") is True
        assert _transcript_contains_transfer("beli nasi") is False

    def test_transcript_looks_noisy(self) -> None:
        assert _transcript_looks_noisy("hmm") is True
        assert _transcript_looks_noisy("eh uh") is True
        assert _transcript_looks_noisy("beli nasi padang 21 ribu") is False


# ────────────────────────────────────────────────────────────────────────────
# Tests: should_trigger
# ────────────────────────────────────────────────────────────────────────────

class TestShouldTrigger:
    def test_disabled_service_never_triggers(self) -> None:
        svc = GroqFallbackService()
        svc._enabled = False
        result = _make_local_result(warnings=["some_warning"])
        assert svc.should_trigger(result) is False

    def test_trigger_on_warnings(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(warnings=["amount_not_detected"])
        assert svc.should_trigger(result) is True

    def test_trigger_on_bad_title(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(title="rp", description="rp")
        assert svc.should_trigger(result) is True

    def test_trigger_on_missing_wallet(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(wallet=None)
        # transcript contains "bca"
        assert svc.should_trigger(result) is True

    def test_trigger_on_inconsistent_transfer(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(
            raw="beli nasi padang",
            normalized="beli nasi padang",
            category="Transfer",
            tx_type="expense",
        )
        assert svc.should_trigger(result) is True

    def test_trigger_on_low_confidence(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(
            wallet="Cash",
            title="Nasi padang",
            description="Nasi padang",
            confidence=0.3,
        )
        assert svc.should_trigger(result) is True

    def test_no_trigger_on_clean_result(self) -> None:
        svc = _make_enabled_service()
        result = _make_local_result(
            raw="beli nasi padang dua puluh satu ribu",
            normalized="beli nasi padang dua puluh satu ribu",
            wallet="Cash",
            title="Nasi padang",
            description="Beli nasi padang",
            warnings=[],
            confidence=0.95,
        )
        assert svc.should_trigger(result) is False


# ────────────────────────────────────────────────────────────────────────────
# Tests: apply_corrections
# ────────────────────────────────────────────────────────────────────────────

class TestApplyCorrections:
    def test_applies_valid_corrections(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result()
        groq_resp = {
            "title": "Nasi Padang",
            "description": "Beli nasi padang",
            "type": "expense",
            "category": "Makan & Minum",
            "wallet": "Cash",
            "reason": "corrected title and wallet",
        }
        result = svc.apply_corrections(local, groq_resp)
        assert result["transaction"]["title"] == "Nasi Padang"
        assert result["transaction"]["description"] == "Beli nasi padang"
        assert result["transaction"]["wallet"] == "Cash"
        assert "groq_fallback_used" in result["warnings"]

    def test_ignores_invalid_category(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result(category="Makan & Minum")
        groq_resp = {
            "title": "Nasi Padang",
            "description": None,
            "type": None,
            "category": "InvalidCategory",
            "wallet": None,
            "reason": "test",
        }
        result = svc.apply_corrections(local, groq_resp)
        # Category should NOT be changed
        assert result["transaction"]["category"] == "Makan & Minum"
        # But title should be applied
        assert result["transaction"]["title"] == "Nasi Padang"
        assert "groq_fallback_used" in result["warnings"]

    def test_ignores_invalid_type(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result(tx_type="expense")
        groq_resp = {
            "title": None,
            "description": None,
            "type": "refund",
            "category": None,
            "wallet": None,
            "reason": "test",
        }
        result = svc.apply_corrections(local, groq_resp)
        assert result["transaction"]["type"] == "expense"
        # Nothing was applied
        assert "groq_fallback_used" not in result["warnings"]

    def test_ignores_invalid_wallet(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result()
        groq_resp = {
            "title": None,
            "description": None,
            "type": None,
            "category": None,
            "wallet": "Tokopedia",
            "reason": "test",
        }
        result = svc.apply_corrections(local, groq_resp)
        assert result["transaction"]["wallet"] is None
        assert "groq_fallback_used" not in result["warnings"]

    def test_does_not_mutate_original(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result()
        original_warnings = list(local["warnings"])
        groq_resp = {
            "title": "New Title",
            "description": None,
            "type": None,
            "category": None,
            "wallet": None,
            "reason": "test",
        }
        svc.apply_corrections(local, groq_resp)
        # Original should not be modified
        assert local["warnings"] == original_warnings
        assert local["transaction"]["title"] is None


# ────────────────────────────────────────────────────────────────────────────
# Tests: maybe_apply (integration with mocked HTTP)
# ────────────────────────────────────────────────────────────────────────────

class TestMaybeApply:
    def test_no_trigger_returns_original(self) -> None:
        svc = _make_enabled_service()
        local = _make_local_result(
            raw="beli nasi padang dua puluh satu ribu",
            normalized="beli nasi padang dua puluh satu ribu",
            wallet="Cash",
            title="Nasi padang",
            description="Beli nasi padang",
            warnings=[],
            confidence=0.95,
        )
        result = svc.maybe_apply(local)
        assert result is local  # same object, not modified

    @patch("app.services.groq_fallback_service.httpx.Client")
    def test_groq_success(self, mock_client_cls: MagicMock) -> None:
        svc = _make_enabled_service()
        local = _make_local_result(warnings=["amount_not_detected"])

        groq_json = {
            "title": "Nasi Padang",
            "description": "Beli nasi padang",
            "type": None,
            "category": None,
            "wallet": "Cash",
            "reason": "corrected",
        }

        # Mock the httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps(groq_json)}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = svc.maybe_apply(local)
        assert result["transaction"]["title"] == "Nasi Padang"
        assert result["transaction"]["wallet"] == "Cash"
        assert "groq_fallback_used" in result["warnings"]

    @patch("app.services.groq_fallback_service.httpx.Client")
    def test_groq_timeout_adds_failed_warning(self, mock_client_cls: MagicMock) -> None:
        import httpx as _httpx

        svc = _make_enabled_service()
        local = _make_local_result(warnings=["amount_not_detected"])

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = _httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        result = svc.maybe_apply(local)
        assert "groq_fallback_failed" in result["warnings"]
        # Original data preserved
        assert result["transaction"]["amount"] == 21000

    @patch("app.services.groq_fallback_service.httpx.Client")
    def test_groq_bad_json_adds_failed_warning(self, mock_client_cls: MagicMock) -> None:
        svc = _make_enabled_service()
        local = _make_local_result(warnings=["amount_not_detected"])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "this is not json"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = svc.maybe_apply(local)
        assert "groq_fallback_failed" in result["warnings"]

    def test_disabled_service_skips_everything(self) -> None:
        svc = GroqFallbackService()
        svc._enabled = False
        local = _make_local_result(warnings=["some_warning"])
        result = svc.maybe_apply(local)
        assert result is local
        assert "groq_fallback_used" not in result.get("warnings", [])
        assert "groq_fallback_failed" not in result.get("warnings", [])


# ────────────────────────────────────────────────────────────────────────────
# Tests: Groq response with markdown code fences
# ────────────────────────────────────────────────────────────────────────────

class TestGroqResponseParsing:
    @patch("app.services.groq_fallback_service.httpx.Client")
    def test_strips_markdown_code_fences(self, mock_client_cls: MagicMock) -> None:
        svc = _make_enabled_service()

        groq_json = {
            "title": "Jeruk Nipis",
            "description": "Beli jeruk nipis",
            "type": "expense",
            "category": "Belanja",
            "wallet": "Cash",
            "reason": "corrected",
        }
        # Groq sometimes wraps in ```json ... ```
        content = f"```json\n{json.dumps(groq_json)}\n```"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        local = _make_local_result(warnings=["amount_not_detected"])
        result = svc.call_groq(local)
        assert result is not None
        assert result["title"] == "Jeruk Nipis"
