"""
Pipeline logging module for structured debugging output with verbosity control.

This module provides the PipelineLogger class for emitting formatted debug logs
at each stage of the transaction parsing pipeline. It supports three verbosity
levels (minimal, standard, verbose) controlled via the PARSER_LOG_LEVEL environment
variable.
"""

import logging
import os
from collections import deque
from typing import Optional


class PipelineLogger:
    """
    Structured logger for the transaction parsing pipeline.
    
    Logs parsing transformations at each stage with configurable verbosity levels:
    - minimal: Only raw input and final prediction
    - standard: All major parsing steps (default)
    - verbose: All standard logs plus intermediate token lists and patterns
    
    The verbosity level is read from the PARSER_LOG_LEVEL environment variable.
    Unknown values default to "standard" with a warning.
    """
    
    def __init__(self):
        """Initialize the logger with verbosity level from environment."""
        raw_level = os.getenv("PARSER_LOG_LEVEL", "standard")
        self._level = raw_level.lower().strip()
        
        # Validate level
        valid_levels = ("minimal", "standard", "verbose")
        if self._level not in valid_levels:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown PARSER_LOG_LEVEL='{raw_level}', defaulting to 'standard'. "
                f"Valid values are: {', '.join(valid_levels)}"
            )
            self._level = "standard"
        
        self._logger = logging.getLogger("fluxa.pipeline")
        self._durations: deque[float] = deque(maxlen=100)
    
    def _emit(self, min_level: str, label: str, *fields: str) -> None:
        """
        Emit a log line if the current verbosity level meets the minimum.
        
        Args:
            min_level: Minimum verbosity level required to emit this log
            label: Log label (e.g., "[RAW INPUT]")
            *fields: Additional field strings to include in the log line
        """
        levels = ("minimal", "standard", "verbose")
        current_idx = levels.index(self._level)
        min_idx = levels.index(min_level)
        
        if current_idx >= min_idx:
            line = " | ".join([label] + list(fields))
            self._logger.debug(line)
    
    def log_raw_input(self, text: str) -> None:
        """
        Log the raw input text before any processing.
        
        Args:
            text: The original raw input text
        """
        self._emit("minimal", "[RAW INPUT]", text)
    
    def log_after_date(
        self,
        date: Optional[str],
        text: str,
        elapsed_ms: float
    ) -> None:
        """
        Log the result after date parsing.
        
        Args:
            date: Extracted date string in YYYY-MM-DD format, or None
            text: Remaining text after date extraction
            elapsed_ms: Parsing duration in milliseconds
        """
        date_str = f"date={date}" if date else "date=None"
        self._emit(
            "standard",
            "[AFTER DATE PARSE]",
            date_str,
            f"text={text}",
            f"elapsed_ms={elapsed_ms:.2f}"
        )
    
    def log_after_amount(
        self,
        amount: Optional[int],
        text: str,
        elapsed_ms: float
    ) -> None:
        """
        Log the result after amount parsing.
        
        Args:
            amount: Extracted amount in rupiah, or None
            text: Remaining text after amount extraction
            elapsed_ms: Parsing duration in milliseconds
        """
        amount_str = f"amount={amount}" if amount is not None else "amount=None"
        self._emit(
            "standard",
            "[AFTER AMOUNT PARSE]",
            amount_str,
            f"text={text}",
            f"elapsed_ms={elapsed_ms:.2f}"
        )
    
    def log_after_title_desc(
        self,
        title: str,
        desc: str,
        elapsed_ms: float
    ) -> None:
        """
        Log the result after title and description extraction.
        
        Args:
            title: Extracted transaction title
            desc: Extracted transaction description
            elapsed_ms: Parsing duration in milliseconds
        """
        self._emit(
            "standard",
            "[AFTER TITLE & DESC PARSE]",
            f"title={title}",
            f"desc={desc}",
            f"elapsed_ms={elapsed_ms:.2f}"
        )
    
    def log_final_prediction(
        self,
        category: str,
        tx_type: str,
        elapsed_ms: float
    ) -> None:
        """
        Log the final classification prediction.
        
        Args:
            category: Predicted transaction category
            tx_type: Predicted transaction type (expense/income)
            elapsed_ms: Classification duration in milliseconds
        """
        self._emit(
            "minimal",
            "[FINAL PREDICTION]",
            f"category={category}",
            f"type={tx_type}",
            f"elapsed_ms={elapsed_ms:.2f}"
        )
    
    def log_duration(self, total_ms: float) -> None:
        """
        Log the total pipeline duration and update rolling average.
        
        Args:
            total_ms: Total pipeline execution time in milliseconds
        """
        self._durations.append(total_ms)
        self._emit(
            "standard",
            "[PIPELINE DURATION]",
            f"total_ms={total_ms:.2f}"
        )
    
    def log_performance_warning(self, total_ms: float) -> None:
        """
        Log a performance warning when pipeline duration exceeds threshold.
        
        Args:
            total_ms: Total pipeline execution time in milliseconds
        """
        self._emit(
            "minimal",
            "[PERFORMANCE WARNING]",
            f"total_ms={total_ms:.2f}",
            "threshold_ms=500"
        )
    
    def log_validation_error(
        self,
        field: str,
        reason: str,
        value: str
    ) -> None:
        """
        Log a validation error when parsed data is invalid.
        
        Args:
            field: The field that failed validation (e.g., "date", "title")
            reason: Description of why validation failed
            value: The invalid value that was rejected
        """
        self._emit(
            "minimal",
            "[VALIDATION ERROR]",
            f"field={field}",
            f"reason={reason}",
            f"value={value}"
        )
    
    @property
    def average_duration_ms(self) -> float:
        """
        Calculate the average pipeline duration over the last 100 transactions.
        
        Returns:
            Average duration in milliseconds, or 0.0 if no data available
        """
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)


# Module-level singleton instance
pipeline_logger = PipelineLogger()
