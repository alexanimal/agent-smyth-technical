"""
Unit tests for RAG utility functions.

This module contains tests for the utility functions used in the
Retrieval Augmented Generation workflow, focusing on caching, LLM generation,
and fallback mechanisms.
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.utils import generate_with_fallback, get_cached_llm


class TestGetCachedLLM:
    """Tests for the get_cached_llm function."""

    def test_allowed_model(self):
        """Test get_cached_llm with an allowed model."""
        # Ensure we have at least one allowed model
        with patch.object(settings, "allowed_models", ["gpt-4", "gpt-3.5-turbo"]):
            with patch("app.rag.utils.ChatOpenAI") as mock_chat_openai:
                # Configure our mock
                mock_instance = Mock(spec=ChatOpenAI)
                mock_chat_openai.return_value = mock_instance

                # Call the function
                result = get_cached_llm("gpt-4")

                # Verify the result and that ChatOpenAI was called correctly
                assert result == mock_instance
                mock_chat_openai.assert_called_once_with(model_name="gpt-4", temperature=0.0)

    def test_disallowed_model_fallback(self):
        """Test get_cached_llm with a disallowed model falls back to default."""
        with patch.object(settings, "allowed_models", ["gpt-4"]):
            with patch.object(settings, "default_model", "gpt-4"):
                with patch("app.rag.utils.ChatOpenAI") as mock_chat_openai:
                    # Configure our mock
                    mock_instance = Mock(spec=ChatOpenAI)
                    mock_chat_openai.return_value = mock_instance

                    # Also patch logger to check warning
                    with patch("app.rag.utils.logger") as mock_logger:
                        # Call the function with a disallowed model
                        result = get_cached_llm("nonexistent-model")

                        # Verify the result
                        assert result == mock_instance

                        # Verify ChatOpenAI was called with the default model
                        mock_chat_openai.assert_called_once_with(
                            model_name="gpt-4", temperature=0.0
                        )

                        # Verify warning was logged
                        mock_logger.warning.assert_called_once()
                        # Check log message contains key information
                        log_args = mock_logger.warning.call_args[0][0]
                        assert "nonexistent-model" in log_args
                        assert "falling back" in log_args.lower()

    def test_lru_cache_behavior(self):
        """Test that the LRU cache properly caches LLM instances."""
        with patch.object(settings, "allowed_models", ["gpt-4", "gpt-3.5-turbo"]):
            with patch("app.rag.utils.ChatOpenAI", wraps=ChatOpenAI) as spy_chat_openai:
                # Reset the cache to ensure clean test state
                get_cached_llm.cache_clear()

                # Call the function twice with the same arguments
                instance1 = get_cached_llm("gpt-4", 0.0)
                instance2 = get_cached_llm("gpt-4", 0.0)

                # Verify ChatOpenAI was called exactly once
                assert spy_chat_openai.call_count == 1

                # Verify both returned instances are the same object
                assert instance1 is instance2

                # Call with different temperature
                instance3 = get_cached_llm("gpt-4", 0.5)

                # Verify ChatOpenAI was called again
                assert spy_chat_openai.call_count == 2

                # Verify the new instance is different
                assert instance1 is not instance3

    def test_custom_temperature(self):
        """Test get_cached_llm with custom temperature."""
        with patch.object(settings, "allowed_models", ["gpt-4"]):
            with patch("app.rag.utils.ChatOpenAI") as mock_chat_openai:
                # Configure our mock
                mock_instance = Mock(spec=ChatOpenAI)
                mock_chat_openai.return_value = mock_instance

                # Call the function with custom temperature
                result = get_cached_llm("gpt-4", temperature=0.7)

                # Verify ChatOpenAI was called with correct temperature
                mock_chat_openai.assert_called_once_with(model_name="gpt-4", temperature=0.7)


class TestGenerateWithFallback:
    """Tests for the generate_with_fallback function."""

    @pytest.mark.asyncio
    async def test_primary_model_success(self):
        """Test successful generation with the primary model."""
        # Setup mocks
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = "Success response"

        # Create a callable that returns our mock
        def get_mock_llm(model_name, temperature):
            return mock_model

        # Mock time to control duration calculation
        with patch("app.rag.utils.get_cached_llm", side_effect=get_mock_llm):
            with patch("app.rag.utils.time.time", side_effect=[0.0, 1.5]):
                with patch("app.rag.utils.logger") as mock_logger:
                    # Call the function
                    result = await generate_with_fallback(
                        prompt="Test prompt", model_name="gpt-4", fallback_model="gpt-3.5-turbo"
                    )

                    # Verify the result
                    assert result == "Success response"

                    # Verify the primary model was called
                    mock_model.ainvoke.assert_called_once_with("Test prompt")

                    # Verify logging
                    mock_logger.info.assert_any_call("Generating with model: gpt-4")
                    # Using a custom matcher for the duration log which may have formatting
                    duration_log_call = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "Generated with gpt-4" in call[0][0]
                    ]
                    assert len(duration_log_call) == 1
                    duration_log = duration_log_call[0][0][0]
                    assert "1.5" in duration_log  # The duration should be in the log

    @pytest.mark.asyncio
    async def test_primary_failure_fallback_success(self):
        """Test fallback to secondary model when primary fails."""
        # Setup mocks for primary (failing) and fallback (succeeding) models
        primary_model = AsyncMock()
        primary_model.ainvoke.side_effect = Exception("Primary model error")

        fallback_model = AsyncMock()
        fallback_model.ainvoke.return_value = "Fallback success"

        # Create a callable that returns the appropriate mock
        def get_mock_llm(model_name, temperature):
            if model_name == "gpt-4":
                return primary_model
            return fallback_model

        # Mock time to control duration calculation - ensure we return enough values
        # for all time.time() calls in the function
        with patch("app.rag.utils.get_cached_llm", side_effect=get_mock_llm):
            # Need at least 4 values: initial, after primary failure, after fallback success, and total duration
            time_values = [0.0, 0.5, 0.8, 1.0]
            with patch("app.rag.utils.time.time", side_effect=time_values):
                with patch("app.rag.utils.logger") as mock_logger:
                    # Call the function
                    result = await generate_with_fallback(
                        prompt="Test prompt", model_name="gpt-4", fallback_model="gpt-3.5-turbo"
                    )

                    # Verify the result
                    assert result == "Fallback success"

                    # Verify both models were called appropriately
                    primary_model.ainvoke.assert_called_once_with("Test prompt")
                    fallback_model.ainvoke.assert_called_once_with("Test prompt")

                    # Verify warning and success logs
                    # Check warning contains key information
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if "failed" in call[0][0] and "gpt-4" in call[0][0]
                    ]
                    assert len(warning_calls) == 1

                    # Check success log
                    success_calls = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "Generated with fallback" in call[0][0]
                    ]
                    assert len(success_calls) == 1

    @pytest.mark.asyncio
    async def test_both_models_fail(self):
        """Test when both primary and fallback models fail."""
        # Setup mocks for both failing models
        primary_model = AsyncMock()
        primary_model.ainvoke.side_effect = ValueError("Primary model error")

        fallback_model = AsyncMock()
        fallback_model.ainvoke.side_effect = RuntimeError("Fallback model error")

        # Create a callable that returns the appropriate mock
        def get_mock_llm(model_name, temperature):
            if model_name == "gpt-4":
                return primary_model
            return fallback_model

        # Mock time to control duration calculation - ensure we return enough values
        # for all time.time() calls in the function
        with patch("app.rag.utils.get_cached_llm", side_effect=get_mock_llm):
            # Need at least 4 values: initial, after primary failure, after fallback failure, and total duration
            time_values = [0.0, 0.5, 0.8, 1.0]
            with patch("app.rag.utils.time.time", side_effect=time_values):
                with patch("app.rag.utils.logger") as mock_logger:
                    # Call the function and expect exception
                    with pytest.raises(Exception) as exc_info:
                        await generate_with_fallback(
                            prompt="Test prompt", model_name="gpt-4", fallback_model="gpt-3.5-turbo"
                        )

                    # Verify exception contains fallback error
                    assert "Failed to generate response" in str(exc_info.value)
                    assert "Fallback model error" in str(exc_info.value)

                    # Verify both models were called
                    primary_model.ainvoke.assert_called_once_with("Test prompt")
                    fallback_model.ainvoke.assert_called_once_with("Test prompt")

                    # Verify error logging
                    mock_logger.error.assert_called()
                    error_log = mock_logger.error.call_args[0][0]
                    assert "Both primary and fallback models failed" in error_log

    @pytest.mark.asyncio
    async def test_custom_temperature(self):
        """Test generate_with_fallback respects custom temperature."""
        # Setup mocks
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = "Success with custom temperature"

        # Create a callable that returns our mock
        with patch("app.rag.utils.get_cached_llm") as mock_get_llm:
            mock_get_llm.return_value = mock_model

            # Mock time.time to avoid StopIteration errors
            with patch("app.rag.utils.time.time", side_effect=[0.0, 1.0]):
                # Call the function with custom temperature
                custom_temp = 0.8
                result = await generate_with_fallback(
                    prompt="Test prompt",
                    model_name="gpt-4",
                    fallback_model="gpt-3.5-turbo",
                    temperature=custom_temp,
                )

                # Verify the result
                assert result == "Success with custom temperature"

                # Verify get_cached_llm was called with correct temperature
                # Check positional and keyword args
                mock_get_llm.assert_called_once()
                args, kwargs = mock_get_llm.call_args
                # Function could be called with positional or keyword args, handle both cases
                if args:
                    # Positional args: get_cached_llm(model_name, temperature)
                    assert len(args) >= 2
                    assert args[0] == "gpt-4"  # First arg is model_name
                    assert args[1] == custom_temp  # Second arg is temperature
                elif "temperature" in kwargs:
                    # Keyword args: get_cached_llm(model_name="gpt-4", temperature=0.8)
                    assert kwargs["temperature"] == custom_temp
                else:
                    # If neither method is used, the test should fail
                    assert (
                        False
                    ), "get_cached_llm was called, but temperature wasn't passed correctly"

    @pytest.mark.asyncio
    async def test_error_logging_detail(self):
        """Test that error details are properly logged when models fail."""
        # Setup mocks for both failing models with detailed error messages
        primary_model = AsyncMock()
        primary_error = ValueError("API rate limit exceeded: 429 Too Many Requests")
        primary_model.ainvoke.side_effect = primary_error

        fallback_model = AsyncMock()
        fallback_error = RuntimeError("Model context length exceeded: maximum 16384 tokens")
        fallback_model.ainvoke.side_effect = fallback_error

        # Create a callable that returns the appropriate mock
        def get_mock_llm(model_name, temperature):
            if model_name == "gpt-4":
                return primary_model
            return fallback_model

        # Mock time for duration calculation
        with patch("app.rag.utils.get_cached_llm", side_effect=get_mock_llm):
            # Provide enough time values for all calls
            time_values = [0.0, 0.5, 0.8, 1.0]
            with patch("app.rag.utils.time.time", side_effect=time_values):
                with patch("app.rag.utils.logger") as mock_logger:
                    # Call the function and expect exception
                    with pytest.raises(Exception) as exc_info:
                        await generate_with_fallback(
                            prompt="Test prompt", model_name="gpt-4", fallback_model="gpt-3.5-turbo"
                        )

                    # Verify exact error message content in exception
                    expected_error = f"Failed to generate response: {str(fallback_error)}"
                    assert str(exc_info.value) == expected_error

                    # Verify detailed error logging for primary model failure
                    warning_log_call = mock_logger.warning.call_args_list[0]
                    warning_msg = warning_log_call[0][0]
                    assert "Model gpt-4 failed" in warning_msg
                    assert "429 Too Many Requests" in warning_msg

                    # Verify detailed error logging for fallback model failure
                    error_log_call = mock_logger.error.call_args_list[0]
                    error_msg = error_log_call[0][0]
                    assert "Both primary and fallback models failed" in error_msg
                    assert "context length exceeded" in str(error_msg)
