"""
Tests for the chat streaming functionality.

These tests validate the Server-Sent Events (SSE) streaming functionality for chat responses,
ensuring that events are correctly formatted and sequenced.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, Header, Request
from sse_starlette.sse import EventSourceResponse

from app.core.handler import ChatHandler
from app.routers.chat_router import stream_chat
from app.schemas import ChatRequest, QueryType

# Test data
TEST_QUERY = "What's happening with AAPL?"
TEST_RESPONSE = {
    "response": "Apple stock has been trending upward in recent weeks due to strong iPhone 15 sales.",
    "sources": ["source1", "source2"],
    "processing_time": 1.24,
    "query_type": "investment",
    "confidence_scores": {"investment": 0.8, "technical": 0.1, "general": 0.1},
}


# Async helper to collect events from a generator
async def collect_events(generator):
    """Collect all events from an async generator."""
    events = []
    try:
        # Convert the generator to an async iterator and collect events
        async for event in generator:
            events.append(event)
    except StopAsyncIteration:
        pass
    except Exception as e:
        print(f"Error collecting events: {e}")
    return events


# Modified test approach that tests the endpoint function directly
@pytest.mark.asyncio
async def test_stream_chat_endpoint():
    """
    Test the stream_chat endpoint function directly by mocking its dependencies.

    This approach bypasses the FastAPI test client and focuses on the endpoint's
    functionality with controlled mocks.
    """
    # Create a mock ChatHandler
    mock_handler = AsyncMock(spec=ChatHandler)
    mock_handler.process_query = AsyncMock(return_value=TEST_RESPONSE)

    # Create a mock request and chat request
    mock_request = MagicMock(spec=Request)
    chat_request = ChatRequest(
        message=TEST_QUERY,
        num_results=3,
        verbose=True,
        query_type=QueryType.INVESTMENT,
        generate_alternative_viewpoint=True,
        ranking_weights={
            "recency_weight": 0.3,
            "view_weight": 0.3,
            "like_weight": 0.2,
            "retweet_weight": 0.2,
        },
    )

    # Create a mock dependency override for get_current_chat_handler
    async def mock_get_handler():
        return mock_handler

    # Store the event generator for later inspection
    captured_generator = None

    # Mock the EventSourceResponse class
    class MockEventSourceResponse:
        def __init__(self, generator, ping=None):
            nonlocal captured_generator
            captured_generator = generator

    # Patch all the necessary dependencies
    with (
        patch("app.routers.chat_router.EventSourceResponse", MockEventSourceResponse),
        patch("app.routers.chat_router.settings.environment", "development"),
        patch("app.routers.chat_router.uuid4", return_value="test-uuid"),
    ):

        # Call the endpoint function directly with mocked headers
        mock_x_request_id = MagicMock(spec=Header)
        mock_x_api_key = MagicMock(spec=Header)

        await stream_chat(
            request_body=chat_request,
            request=mock_request,
            x_request_id="test-request-id",
            x_api_key="test-api-key",
            chat_service=await mock_get_handler(),
        )

        # Verify the generator was captured
        assert captured_generator is not None

        # Collect and check events
        events = await collect_events(captured_generator)

        # Verify we got the expected events
        assert len(events) >= 3  # At least start, one or more chunks, and complete

        # Check event types
        event_types = [event.get("event") for event in events]
        assert "start" in event_types
        assert "chunk" in event_types
        assert "complete" in event_types

        # Verify the handler was called with correct arguments
        from unittest.mock import ANY

        mock_handler.process_query.assert_called_once_with(
            message=TEST_QUERY,
            k=3,
            ranking_weights={
                "recency_weight": 0.3,
                "view_weight": 0.3,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            },
            model=ANY,
            context=ANY,
            generate_alternative_viewpoint=True,
        )

        # Verify the complete event has the expected data
        complete_event = [e for e in events if e.get("event") == "complete"][0]
        complete_data = json.loads(complete_event["data"])
        assert complete_data["response"] == TEST_RESPONSE["response"]
        assert complete_data["sources"] == TEST_RESPONSE["sources"]


@pytest.mark.asyncio
async def test_stream_chat_error_handling():
    """
    Test error handling in the stream_chat endpoint function.
    """
    # Create a mock ChatHandler that raises an exception
    mock_handler = AsyncMock(spec=ChatHandler)
    mock_handler.process_query = AsyncMock(side_effect=ValueError("Test error"))

    # Create a mock request and chat request
    mock_request = MagicMock(spec=Request)
    chat_request = ChatRequest(
        message=TEST_QUERY,
        num_results=3,
        query_type=QueryType.INVESTMENT,
        verbose=True,
        generate_alternative_viewpoint=False,
        ranking_weights={
            "recency_weight": 0.3,
            "view_weight": 0.3,
            "like_weight": 0.2,
            "retweet_weight": 0.2,
        },
    )

    # Store the event generator for later inspection
    captured_generator = None

    # Mock the EventSourceResponse class
    class MockEventSourceResponse:
        def __init__(self, generator, ping=None):
            nonlocal captured_generator
            captured_generator = generator

    # Patch all the necessary dependencies
    with (
        patch("app.routers.chat_router.EventSourceResponse", MockEventSourceResponse),
        patch("app.routers.chat_router.settings.environment", "development"),
        patch("app.routers.chat_router.uuid4", return_value="test-uuid"),
    ):

        # Call the endpoint function directly with mocked headers
        await stream_chat(
            request_body=chat_request,
            request=mock_request,
            x_request_id="test-request-id",
            x_api_key="test-api-key",
            chat_service=mock_handler,
        )

        # Verify the generator was captured
        assert captured_generator is not None

        # Collect and check events
        events = await collect_events(captured_generator)

        # Verify we got the start and error events
        assert len(events) == 2

        # Check event types
        event_types = [event.get("event") for event in events]
        assert "start" in event_types
        assert "error" in event_types

        # Verify the handler was called
        mock_handler.process_query.assert_called_once()

        # Verify the error event has the expected data
        error_event = [e for e in events if e.get("event") == "error"][0]
        error_data = json.loads(error_event["data"])
        assert error_data["status"] == "error"
        assert "Test error" in error_data["error"]
