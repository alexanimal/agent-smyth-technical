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


@pytest.fixture(scope="session", autouse=True)
def mock_env_vars(monkeypatch):
    """Automatically mock required environment variables for the test session."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    # Add any other required environment variables here if needed
    # monkeypatch.setenv("ANOTHER_VAR", "dummy_value")
