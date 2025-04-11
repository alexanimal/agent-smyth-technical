"""
Configuration package for the RAG Agent application.

This package re-exports the Settings class and other configuration elements
from the settings module to maintain backward compatibility.
"""

from app.config.settings import (
    MOCKS_DIR_PATH,
    PROJECT_ROOT,
    Settings,
    settings,
    setup_logging_and_sentry,
)

# For backward compatibility
__all__ = ["settings", "Settings", "setup_logging_and_sentry", "PROJECT_ROOT", "MOCKS_DIR_PATH"]
