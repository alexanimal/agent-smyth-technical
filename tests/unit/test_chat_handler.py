"""
Tests for the ChatHandler class.

This module contains tests for the ChatHandler class which manages the RAG pipeline
for processing user queries, including document retrieval, re-ranking,
prompt selection, and response generation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from app.core.handler import ChatHandler
from app.core.router import ChatRouter
from app.rag.graph import app_workflow


class TestChatHandler:
    """Tests for the ChatHandler class."""

    @pytest.fixture
    def mock_vector_store(self) -> MagicMock:
        """Create a mock VectorStore."""
        vector_store = MagicMock(spec=VectorStore)
        return vector_store

    @pytest.fixture
    def chat_handler(self, mock_vector_store) -> ChatHandler:
        """Create a ChatHandler instance with mock dependencies."""
        handler = ChatHandler(
            knowledge_base=mock_vector_store,
            model_name="gpt-4o",
            temperature=0.0,
            max_retries=2,
            retry_delay=0.01,  # Very short delay for testing
        )
        return handler

    def test_initialization(self, chat_handler, mock_vector_store):
        """Test the initialization of ChatHandler with proper parameters."""
        assert chat_handler.knowledge_base == mock_vector_store
        assert chat_handler.model_name == "gpt-4o"
        assert chat_handler.temperature == 0.0
        assert chat_handler.max_retries == 2
        assert chat_handler.retry_delay == 0.01
        assert chat_handler._llm is None
        assert chat_handler._router is None
        assert chat_handler._technical_llm is None
        assert chat_handler._model_instances == {}

    def test_llm_property(self, chat_handler):
        """Test the lazy-loaded llm property."""
        # First access should create the model
        with patch("app.core.handler.ChatOpenAI") as mock_chat_openai:
            mock_model = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_model

            # Access the property
            result = chat_handler.llm

            # Verify the model was created with the correct parameters
            mock_chat_openai.assert_called_once_with(model_name="gpt-4o", temperature=0.0)
            assert result == mock_model

            # Access again, should use cached version
            chat_handler.llm
            # Still only called once
            mock_chat_openai.assert_called_once()

    def test_explorer_llm_property(self, chat_handler):
        """Test the lazy-loaded explorer_llm property."""
        # First access should create the model
        with patch("app.core.handler.ChatOpenAI") as mock_chat_openai:
            mock_model = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_model

            # Access the property
            result = chat_handler.explorer_llm

            # Verify the model was created with the correct parameters
            # Should have higher temperature for exploration
            mock_chat_openai.assert_called_once_with(
                model_name="gpt-4o", temperature=0.3  # Base 0.0 + 0.3
            )
            assert result == mock_model

    def test_technical_llm_property(self, chat_handler):
        """Test the lazy-loaded technical_llm property."""
        # First access should create the model
        with patch("app.core.handler.ChatOpenAI") as mock_chat_openai:
            mock_model = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_model

            # Access the property
            result = chat_handler.technical_llm

            # Verify the model was created with the correct parameters
            # Should have lower temperature for technical analysis
            mock_chat_openai.assert_called_once_with(
                model_name="gpt-4o", temperature=0.0  # Base 0.0, but min is 0
            )
            assert result == mock_model

    def test_router_property(self, chat_handler):
        """Test the lazy-loaded router property."""
        # First access should create the router
        with (
            patch("app.core.handler.ChatOpenAI") as mock_chat_openai,
            patch("app.core.handler.ChatRouter") as mock_chat_router,
        ):

            # Configure mocks
            mock_model = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_model

            mock_router = MagicMock(spec=ChatRouter)
            mock_chat_router.return_value = mock_router

            # Access the property
            result = chat_handler.router

            # Verify the router was created with the correct parameters
            mock_chat_openai.assert_called_once_with(model_name="gpt-3.5-turbo", temperature=0.0)
            mock_chat_router.assert_called_once_with(mock_model)

    def test_prompt_getters(self, chat_handler):
        """Test the prompt getter methods."""
        with patch("app.core.handler.PromptManager") as mock_prompt_manager:
            # Configure mocks for each prompt type
            mock_investment_prompt = MagicMock()
            mock_general_prompt = MagicMock()
            mock_trading_thesis_prompt = MagicMock()
            mock_technical_analysis_prompt = MagicMock()

            mock_prompt_manager.get_investment_prompt.return_value = mock_investment_prompt
            mock_prompt_manager.get_general_prompt.return_value = mock_general_prompt
            mock_prompt_manager.get_trading_thesis_prompt.return_value = mock_trading_thesis_prompt
            mock_prompt_manager.get_technical_analysis_prompt.return_value = (
                mock_technical_analysis_prompt
            )

            # Test each prompt getter
            assert chat_handler._get_investment_prompt() == mock_investment_prompt
            assert chat_handler._get_general_prompt() == mock_general_prompt
            assert chat_handler._get_trading_thesis_prompt() == mock_trading_thesis_prompt
            assert chat_handler._get_technical_analysis_prompt() == mock_technical_analysis_prompt

            # Verify each method was called once
            mock_prompt_manager.get_investment_prompt.assert_called_once()
            mock_prompt_manager.get_general_prompt.assert_called_once()
            mock_prompt_manager.get_trading_thesis_prompt.assert_called_once()
            mock_prompt_manager.get_technical_analysis_prompt.assert_called_once()

    def test_extract_sources(self, chat_handler):
        """Test the _extract_sources method for extracting URLs from document metadata."""
        # Create mock documents with URLs
        from langchain_core.documents import Document

        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={"url": "https://example.com/2"}),
            Document(page_content="Doc 3", metadata={"url": "https://example.com/1"}),  # Duplicate
        ]

        # Call the method with mocked utils.document.extract_sources
        with patch("app.core.handler.extract_sources") as mock_extract_sources:
            mock_extract_sources.return_value = ["https://example.com/1", "https://example.com/2"]

            result = chat_handler._extract_sources(docs)

            # Verify extract_sources was called and the result is correct
            mock_extract_sources.assert_called_once_with(docs)
            assert result == ["https://example.com/1", "https://example.com/2"]

    @pytest.mark.asyncio
    async def test_process_query_success(self, chat_handler):
        """Test the process_query method with successful execution."""
        # Mock data
        test_query = "What's the current price of AAPL?"
        expected_response = {
            "response": "AAPL is currently trading at $200.",
            "sources": ["https://example.com/1", "https://example.com/2"],
            "processing_time": 1.0,
            "query_type": "investment",
            "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
        }

        # Create mock for app_workflow
        mock_workflow_result = {
            "response": expected_response["response"],
            "sources": expected_response["sources"],
            "classification": {
                "query_type": "investment",
                "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
            },
        }

        # Patch dependencies
        with (
            patch("app.core.handler.app_workflow.ainvoke") as mock_workflow_invoke,
            patch("app.core.handler.time.time") as mock_time,
            patch(
                "app.core.handler.logger"
            ) as mock_logger,  # Mock the logger to avoid time.time() calls in logging
        ):
            # Configure mocks
            # Define an async function to return the mock result
            async def mock_ainvoke(*args, **kwargs):
                return mock_workflow_result

            mock_workflow_invoke.side_effect = mock_ainvoke

            # Use a function for time.time to always return increasing values
            time_values = [0.0, 1.0]
            mock_time.side_effect = lambda: (
                time_values[0]
                if not hasattr(mock_time, "call_count") or mock_time.call_count < 2
                else time_values[1]
            )

            # Call the method
            result = await chat_handler.process_query(test_query, k=5)

            # Verify workflow was called with correct initial state
            mock_workflow_invoke.assert_called_once()
            call_args = mock_workflow_invoke.call_args[0][0]
            assert call_args["query"] == test_query
            assert "retrieved_docs" in call_args
            assert "ranked_docs" in call_args

            # Verify the result is correct
            assert result["response"] == expected_response["response"]
            assert result["sources"] == expected_response["sources"]
            assert result["processing_time"] == 1.0
            assert result["query_type"] == "investment"
            assert result["confidence_scores"] == {
                "investment": 0.8,
                "technical": 0.1,
                "general": 0.1,
            }

    @pytest.mark.asyncio
    async def test_process_query_with_alternative_viewpoints(self, chat_handler):
        """Test the process_query method with alternative viewpoints in the response."""
        # Mock data
        test_query = "What's the current price of AAPL?"
        alternative_view = (
            "While many analysts are bullish on AAPL, there are concerns about market saturation."
        )

        # Create mock for app_workflow with alternative viewpoints
        mock_workflow_result = {
            "response": "AAPL is currently trading at $200.",
            "sources": ["https://example.com/1", "https://example.com/2"],
            "classification": {
                "query_type": "investment",
                "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
            },
            "alternative_viewpoints": alternative_view,
        }

        # Patch dependencies
        with (
            patch(
                "app.core.handler.app_workflow.ainvoke", new_callable=AsyncMock
            ) as mock_workflow_invoke,
            patch("app.core.handler.time.time"),
        ):

            # Configure mocks
            mock_workflow_invoke.return_value = mock_workflow_result

            # Call the method with generate_alternative_viewpoint set to True
            result = await chat_handler.process_query(
                test_query, generate_alternative_viewpoint=True
            )

            # Verify app_workflow was called with the correct state
            mock_workflow_invoke.assert_called_once()
            initial_state = mock_workflow_invoke.call_args[0][0]
            assert initial_state["generate_alternative_viewpoint"] is True

            # Verify alternative viewpoints are included in the result
            assert "alternative_viewpoints" in result
            assert result["alternative_viewpoints"] == alternative_view

            # Reset the mock
            mock_workflow_invoke.reset_mock()

            # Now call without the flag and verify it's set to False
            result = await chat_handler.process_query(
                test_query  # generate_alternative_viewpoint defaults to False
            )

            # Verify app_workflow was called with generate_alternative_viewpoint=False
            mock_workflow_invoke.assert_called_once()
            initial_state = mock_workflow_invoke.call_args[0][0]
            assert initial_state["generate_alternative_viewpoint"] is False

    @pytest.mark.asyncio
    async def test_process_query_error_retry(self, chat_handler):
        """Test the process_query method with an error that triggers a retry."""
        # Mock data
        test_query = "What's the current price of AAPL?"
        expected_response = {
            "response": "AAPL is currently trading at $200.",
            "sources": ["https://example.com/1"],
            "processing_time": 1.0,
            "query_type": "investment",
            "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
        }

        # Create mock for app_workflow that fails once then succeeds
        mock_workflow_result = {
            "response": expected_response["response"],
            "sources": expected_response["sources"],
            "classification": {
                "query_type": "investment",
                "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
            },
        }

        # Patch dependencies and track call count
        attempt_count = 0

        async def mock_async_invoke(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            # First call raises error, subsequent calls succeed
            if attempt_count == 1:
                raise ValueError("Test error")
            return mock_workflow_result

        # Use context managers for patching
        with (
            patch("app.core.handler.app_workflow") as mock_workflow,
            patch("app.core.handler.time.time", return_value=123),
            patch("app.core.handler.asyncio.sleep", new_callable=AsyncMock),
        ):
            # Setup the mock with our custom side effect
            mock_workflow.ainvoke = AsyncMock(side_effect=mock_async_invoke)

            # Call the method with retry
            result = await chat_handler.process_query(test_query)

            # Verify the workflow was invoked the expected number of times
            assert attempt_count == 2  # Initial call + 1 retry

            # Verify the method was called at least twice (once failing, once succeeding)
            assert mock_workflow.ainvoke.await_count >= 2

            # Verify the result has the expected values
            assert result["response"] == expected_response["response"]
            assert result["sources"] == expected_response["sources"]
            assert result["query_type"] == "investment"

    @pytest.mark.asyncio
    async def test_process_query_max_retries_exceeded(self, chat_handler):
        """Test the process_query method when max retries are exceeded."""
        # Mock data
        test_query = "What's the current price of AAPL?"
        test_error = ValueError("Test error")

        # Set max retries to a smaller value to speed up testing
        original_max_retries = chat_handler.max_retries
        chat_handler.max_retries = 2

        # Keep track of how many times our function is called
        attempt_count = 0

        async def always_raise_error(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise test_error

        # Patch dependencies
        try:
            with (
                patch("app.core.handler.app_workflow") as mock_workflow,
                patch("app.core.handler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                # Setup the mock to always raise an error with our tracking function
                mock_workflow.ainvoke = AsyncMock(side_effect=always_raise_error)

                # Call the method and expect an error
                with pytest.raises(ValueError) as exc_info:
                    await chat_handler.process_query(test_query)

                # Verify the correct error was raised
                assert str(exc_info.value) == "Test error"

                # Verify sleep was called for each retry (not for the initial attempt)
                assert mock_sleep.await_count == chat_handler.max_retries

                # Verify workflow was called the expected number of times:
                # 1 initial attempt + max_retries
                assert attempt_count == chat_handler.max_retries + 1
        finally:
            # Restore original max_retries to avoid affecting other tests
            chat_handler.max_retries = original_max_retries

    @pytest.mark.asyncio
    async def test_process_query_different_error_types(self, chat_handler):
        """Test the process_query method with different types of errors."""
        # Mock data
        test_query = "What's the current price of AAPL?"

        # Set max retries to match our error sequence length minus 1
        # This way we retry exactly as many times as needed to reach the last error
        original_max_retries = chat_handler.max_retries
        chat_handler.max_retries = 2  # 3 errors total - 1 initial attempt + 2 retries

        # Setup the error sequence
        error_sequence = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
        ]

        # Counter to track which error to raise
        attempt_count = 0

        async def sequence_of_errors(*args, **kwargs):
            nonlocal attempt_count
            if attempt_count < len(error_sequence):
                error = error_sequence[attempt_count]
                attempt_count += 1
                raise error
            # This should never be reached in this test
            return {}

        # Patch dependencies
        try:
            with (
                patch("app.core.handler.app_workflow") as mock_workflow,
                patch("app.core.handler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                # Setup the mock with our sequence of errors
                mock_workflow.ainvoke = AsyncMock(side_effect=sequence_of_errors)

                # Call the method and expect the RuntimeError (the last in our sequence)
                with pytest.raises(RuntimeError) as exc_info:
                    await chat_handler.process_query(test_query)

                # Verify the error from the last attempt was raised
                assert str(exc_info.value) == "Runtime error"

                # Verify that sleep was called between each retry (twice for 3 errors)
                assert mock_sleep.await_count == 2

                # Verify we attempted exactly as many times as there are errors
                assert attempt_count == 3
        finally:
            # Restore original max_retries to avoid affecting other tests
            chat_handler.max_retries = original_max_retries
