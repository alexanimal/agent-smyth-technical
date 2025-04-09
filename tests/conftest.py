"""
Configuration file for pytest.
This file is automatically loaded by pytest and can be used to define fixtures and setup code.
"""

import os
import sys

import pytest

# Add the project root directory to the Python path
# This ensures that imports from the project root work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# Register pytest markers
def pytest_configure(config):
    """Register custom markers for categorizing tests."""
    config.addinivalue_line("markers", "basic: mark a test as a basic test")
    config.addinivalue_line("markers", "advanced: mark a test as an advanced test")


@pytest.fixture(scope="function", autouse=True)
def mock_env_vars(monkeypatch):
    """Automatically mock required environment variables for the test session."""
    monkeypatch.setenv("OPENAI_API_KEY", "1234567890")
    # Add any other required environment variables here if needed
    # monkeypatch.setenv("ANOTHER_VAR", "dummy_value")
