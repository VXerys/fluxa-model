"""
Unit tests for the PipelineLogger class.

Tests verbosity levels, log formatting, environment variable handling,
and rolling average calculation.
"""

import logging
import os
import pytest
from unittest.mock import patch

from src.pipeline_logger import PipelineLogger, pipeline_logger


class TestPipelineLoggerVerbosity:
    """Test verbosity level handling and filtering."""
    
    def test_minimal_level_only_shows_minimal_logs(self, caplog):
        """WHEN verbosity is 'minimal', THEN only minimal logs are emitted."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "minimal"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_raw_input("test input")
                logger.log_final_prediction("Makan & Minum", "expense", 5.0)
                logger.log_after_date("2026-07-04", "remaining text", 2.0)
                logger.log_duration(10.0)
                
            # Should have 2 log entries (raw input, final prediction)
            assert len(caplog.records) == 2
            assert "[RAW INPUT]" in caplog.text
            assert "[FINAL PREDICTION]" in caplog.text
            assert "[AFTER DATE PARSE]" not in caplog.text
            assert "[PIPELINE DURATION]" not in caplog.text
    
    def test_standard_level_shows_standard_and_minimal(self, caplog):
        """WHEN verbosity is 'standard', THEN standard and minimal logs are emitted."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_raw_input("test input")
                logger.log_after_date("2026-07-04", "remaining", 2.0)
                logger.log_after_amount(25000, "text", 3.0)
                logger.log_after_title_desc("kopi", "buat lembur", 1.0)
                logger.log_final_prediction("Makan & Minum", "expense", 5.0)
                logger.log_duration(11.0)
                
            assert len(caplog.records) == 6
            assert "[RAW INPUT]" in caplog.text
            assert "[AFTER DATE PARSE]" in caplog.text
            assert "[AFTER AMOUNT PARSE]" in caplog.text
            assert "[AFTER TITLE & DESC PARSE]" in caplog.text
            assert "[FINAL PREDICTION]" in caplog.text
            assert "[PIPELINE DURATION]" in caplog.text
    
    def test_verbose_level_shows_all_logs(self, caplog):
        """WHEN verbosity is 'verbose', THEN all logs including verbose are emitted."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "verbose"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_raw_input("test")
                logger.log_final_prediction("Cat", "expense", 1.0)
                logger.log_after_date("2026-07-04", "text", 2.0)
                
            # All logs should be present
            assert "[RAW INPUT]" in caplog.text
            assert "[FINAL PREDICTION]" in caplog.text
            assert "[AFTER DATE PARSE]" in caplog.text


class TestPipelineLoggerEnvironment:
    """Test environment variable handling."""
    
    def test_default_to_standard_when_env_not_set(self):
        """WHEN PARSER_LOG_LEVEL is not set, THEN default to 'standard'."""
        with patch.dict(os.environ, {}, clear=True):
            logger = PipelineLogger()
            assert logger._level == "standard"
    
    def test_unknown_level_defaults_to_standard_with_warning(self, caplog):
        """WHEN PARSER_LOG_LEVEL is unknown, THEN default to 'standard' and warn."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "invalid"}):
            with caplog.at_level(logging.WARNING):
                logger = PipelineLogger()
                
            assert logger._level == "standard"
            assert "Unknown PARSER_LOG_LEVEL" in caplog.text
            assert "defaulting to 'standard'" in caplog.text
    
    def test_case_insensitive_level_parsing(self):
        """WHEN PARSER_LOG_LEVEL has mixed case, THEN normalize to lowercase."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "MINIMAL"}):
            logger = PipelineLogger()
            assert logger._level == "minimal"
        
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "  Standard  "}):
            logger = PipelineLogger()
            assert logger._level == "standard"
    
    def test_whitespace_stripped_from_level(self):
        """WHEN PARSER_LOG_LEVEL has whitespace, THEN strip it."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "  verbose  "}):
            logger = PipelineLogger()
            assert logger._level == "verbose"


class TestPipelineLoggerFormatting:
    """Test log message formatting."""
    
    def test_raw_input_format(self, caplog):
        """Verify raw input log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "minimal"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_raw_input("beli kopi kemaren 25rb")
                
            assert "[RAW INPUT] | beli kopi kemaren 25rb" in caplog.text
    
    def test_after_date_format(self, caplog):
        """Verify date parsing log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_date("2026-07-04", "beli kopi 25rb", 1.234)
                
            log_line = caplog.text
            assert "[AFTER DATE PARSE]" in log_line
            assert "date=2026-07-04" in log_line
            assert "text=beli kopi 25rb" in log_line
            assert "elapsed_ms=1.23" in log_line
    
    def test_after_date_format_with_none(self, caplog):
        """Verify date parsing log format when date is None."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_date(None, "beli kopi 25rb", 1.5)
                
            log_line = caplog.text
            assert "[AFTER DATE PARSE]" in log_line
            assert "date=None" in log_line
    
    def test_after_amount_format(self, caplog):
        """Verify amount parsing log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_amount(25000, "beli kopi", 2.567)
                
            log_line = caplog.text
            assert "[AFTER AMOUNT PARSE]" in log_line
            assert "amount=25000" in log_line
            assert "text=beli kopi" in log_line
            assert "elapsed_ms=2.57" in log_line
    
    def test_after_amount_format_with_none(self, caplog):
        """Verify amount parsing log format when amount is None."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_amount(None, "beli kopi", 2.0)
                
            log_line = caplog.text
            assert "amount=None" in log_line
    
    def test_after_title_desc_format(self, caplog):
        """Verify title and description log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_title_desc("kopi kenangan", "buat lembur", 3.45)
                
            log_line = caplog.text
            assert "[AFTER TITLE & DESC PARSE]" in log_line
            assert "title=kopi kenangan" in log_line
            assert "desc=buat lembur" in log_line
            assert "elapsed_ms=3.45" in log_line
    
    def test_final_prediction_format(self, caplog):
        """Verify final prediction log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "minimal"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_final_prediction("Makan & Minum", "expense", 4.12)
                
            log_line = caplog.text
            assert "[FINAL PREDICTION]" in log_line
            assert "category=Makan & Minum" in log_line
            assert "type=expense" in log_line
            assert "elapsed_ms=4.12" in log_line
    
    def test_duration_format(self, caplog):
        """Verify pipeline duration log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_duration(123.456)
                
            log_line = caplog.text
            assert "[PIPELINE DURATION]" in log_line
            assert "total_ms=123.46" in log_line
    
    def test_performance_warning_format(self, caplog):
        """Verify performance warning log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "minimal"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_performance_warning(623.789)
                
            log_line = caplog.text
            assert "[PERFORMANCE WARNING]" in log_line
            assert "total_ms=623.79" in log_line
            assert "threshold_ms=500" in log_line
    
    def test_validation_error_format(self, caplog):
        """Verify validation error log format."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "minimal"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_validation_error(
                    "date",
                    "outside 365-day window",
                    "2019-01-01"
                )
                
            log_line = caplog.text
            assert "[VALIDATION ERROR]" in log_line
            assert "field=date" in log_line
            assert "reason=outside 365-day window" in log_line
            assert "value=2019-01-01" in log_line


class TestPipelineLoggerAverageDuration:
    """Test rolling average duration calculation."""
    
    def test_average_duration_empty_initially(self):
        """WHEN no durations logged, THEN average is 0.0."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            assert logger.average_duration_ms == 0.0
    
    def test_average_duration_single_value(self):
        """WHEN one duration logged, THEN average equals that value."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            logger.log_duration(50.0)
            assert logger.average_duration_ms == 50.0
    
    def test_average_duration_multiple_values(self):
        """WHEN multiple durations logged, THEN average is correct."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            logger.log_duration(10.0)
            logger.log_duration(20.0)
            logger.log_duration(30.0)
            
            expected_avg = (10.0 + 20.0 + 30.0) / 3
            assert logger.average_duration_ms == expected_avg
    
    def test_average_duration_rolling_window_maxlen_100(self):
        """WHEN more than 100 durations logged, THEN only last 100 are used."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            # Log 150 values
            for i in range(150):
                logger.log_duration(float(i))
            
            # Should only have last 100 values (50-149)
            expected_avg = sum(range(50, 150)) / 100
            assert logger.average_duration_ms == expected_avg
    
    def test_average_duration_property_read_only(self):
        """Verify average_duration_ms is a property."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            logger.log_duration(25.0)
            
            # Should be accessible as a property
            avg = logger.average_duration_ms
            assert avg == 25.0


class TestModuleLevelSingleton:
    """Test the module-level pipeline_logger singleton."""
    
    def test_pipeline_logger_singleton_exists(self):
        """Verify pipeline_logger singleton is exported."""
        assert pipeline_logger is not None
        assert isinstance(pipeline_logger, PipelineLogger)
    
    def test_pipeline_logger_singleton_is_functional(self, caplog):
        """Verify the singleton can log messages."""
        with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
            pipeline_logger.log_raw_input("singleton test")
        
        assert "[RAW INPUT]" in caplog.text


class TestFieldSeparator:
    """Test that fields are separated by ' | '."""
    
    def test_fields_separated_by_pipe(self, caplog):
        """Verify fields are joined with ' | ' separator."""
        with patch.dict(os.environ, {"PARSER_LOG_LEVEL": "standard"}):
            logger = PipelineLogger()
            
            with caplog.at_level(logging.DEBUG, logger="fluxa.pipeline"):
                logger.log_after_date("2026-07-04", "remaining text", 2.5)
                
            # Check for proper field separation
            assert " | " in caplog.text
            parts = [line for line in caplog.text.split("\n") if "[AFTER DATE PARSE]" in line][0]
            assert parts.count(" | ") >= 3  # At least 3 separators
