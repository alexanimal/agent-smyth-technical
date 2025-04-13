"""
Unit tests for settings module.

This module tests the setup_logging_and_sentry function in the settings module.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import sentry_sdk

# Import the function directly to avoid triggering Settings initialization
import app.config.settings
from app.config.settings import setup_logging_and_sentry


class TestLoggingAndSentry:
    """Tests for the setup_logging_and_sentry function."""

    def test_logging_setup(self):
        """Test that logging is configured with the correct level."""
        with patch("logging.basicConfig") as mock_basic_config:
            mock_settings = MagicMock()
            mock_settings.log_level = "DEBUG"
            mock_settings.sentry_dsn = None

            setup_logging_and_sentry(mock_settings)

            mock_basic_config.assert_called_once_with(level="DEBUG")

    def test_sentry_init_production(self):
        """Test that Sentry is initialized in production environment."""
        with patch("sentry_sdk.init") as mock_sentry_init:
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                mock_settings = MagicMock()
                mock_settings.sentry_dsn = "https://sentry.example.com"
                mock_settings.environment = "production"

                setup_logging_and_sentry(mock_settings)

                mock_sentry_init.assert_called_once()
                mock_logger.info.assert_called_with("Sentry initialized.")

    def test_sentry_skip_non_production(self):
        """Test that Sentry initialization is skipped in non-production environments."""
        with patch("sentry_sdk.init") as mock_sentry_init:
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                mock_settings = MagicMock()
                mock_settings.sentry_dsn = "https://sentry.example.com"
                mock_settings.environment = "development"

                setup_logging_and_sentry(mock_settings)

                mock_sentry_init.assert_not_called()
                mock_logger.warning.assert_called_once()

    def test_sentry_skip_no_dsn(self):
        """Test that Sentry initialization is skipped when no DSN is provided."""
        with patch("sentry_sdk.init") as mock_sentry_init:
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                mock_settings = MagicMock()
                mock_settings.sentry_dsn = None
                mock_settings.environment = "production"

                setup_logging_and_sentry(mock_settings)

                mock_sentry_init.assert_not_called()
                mock_logger.info.assert_called_with("Sentry DSN not found. Sentry not initialized.")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
