from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.inference_pipeline import infer_transaction


class TransactionParserService:
    def __init__(self) -> None:
        # File ini ada di:
        # ai_backend/app/services/transaction_parser_service.py
        # parents[3] = root project fluxa_voice_ai_training_pipeline
        self.project_root = Path(__file__).resolve().parents[2]

        self.type_model_path = (
            self.project_root / "models" / "baseline_type_classifier.joblib"
        )
        self.category_model_path = (
            self.project_root / "models" / "baseline_category_classifier.joblib"
        )

        if not self.type_model_path.exists():
            raise FileNotFoundError(f"Type model not found: {self.type_model_path}")

        if not self.category_model_path.exists():
            raise FileNotFoundError(
                f"Category model not found: {self.category_model_path}"
            )

        self.type_model: Any = joblib.load(self.type_model_path)
        self.category_model: Any = joblib.load(self.category_model_path)

    def parse_text(self, text: str) -> dict[str, Any]:
        return infer_transaction(
            text=text,
            type_model=self.type_model,
            category_model=self.category_model,
        )


transaction_parser_service = TransactionParserService()
