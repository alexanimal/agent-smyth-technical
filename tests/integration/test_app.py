"""
Integration tests for the FastAPI application in main.py
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Create a patch for the KnowledgeBaseManager before importing the app
with (
    patch("app.services.KnowledgeBaseManager") as mock_kb_manager,
    patch("app.services.ChatHandler") as mock_chat_handler,
):
    # Now import the app after patching
    from app.main import app

# Test client
client = TestClient(app)


@pytest.fixture
def mock_knowledge_base():
    """Create a mock knowledge base."""
    return MagicMock()


@pytest.fixture
def mock_chat_response():
    """Create a mock response from the chat handler."""
    return {
        "response": "This is a test response",
        "sources": ["https://example.com/1", "https://example.com/2"],
        "processing_time": 0.5,
        "query_type": "general",
    }


@pytest.fixture
def setup_mocks(mock_knowledge_base, mock_chat_response):
    """Setup all necessary mocks."""
    # Mock the application state in services.py
    from app import services

    services.app_state["knowledge_base"] = mock_knowledge_base
    services.app_state["is_kb_loading"] = False

    # Create a mock chat handler and place it in the app_state
    mock_handler = AsyncMock()
    mock_handler.process_query = AsyncMock(return_value=mock_chat_response)
    services.app_state["chat_handler"] = mock_handler

    return mock_handler


def test_root_endpoint(setup_mocks):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "ready"  # Since we mocked app_state


def test_health_check(setup_mocks):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "knowledge_base_loaded": True,  # Based on mocked app_state
        "is_loading": False,
    }


def test_chat_endpoint(setup_mocks):
    """Test the chat endpoint."""
    # The dependency `get_current_chat_handler` now reads from services.app_state
    # which we already mocked in setup_mocks. No need to patch the dependency getter itself.
    response = client.post(
        "/chat",
        headers={"User-Agent": "pytest-client"},  # Add headers if middleware uses them
        json={"message": "Test message", "num_results": 3},
    )

    # Check response
    assert response.status_code == 200
    response_data = response.json()
    # Assert individual keys that are expected to be static
    assert response_data["response"] == "This is a test response"
    assert response_data["sources"] == ["https://example.com/1", "https://example.com/2"]
    # Optionally check processing_time if the mock provides it consistently
    # assert response_data["processing_time"] == 0.5
    # Check existence/type of dynamic keys
    assert "request_id" in response_data
    assert isinstance(response_data["request_id"], str)
    assert "timestamp" in response_data
    assert "metadata" in response_data  # Check metadata structure if needed

    # Verify the mock handler in app_state was called correctly
    setup_mocks.process_query.assert_called_once_with(message="Test message", k=3)


def test_chat_endpoint_error(setup_mocks):
    """Test error handling in the chat endpoint."""
    # Make the handler in app_state raise an exception
    setup_mocks.process_query.side_effect = Exception("Test error")

    # No need to patch the dependency getter
    response = client.post("/chat", json={"message": "Test message"})

    # Check error response
    assert response.status_code == 500
    # Check specific detail if needed, depends on router's exception handling
    assert "internal error" in response.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_initialize_services_success():
    """Test successful service initialization."""
    # Mock dependencies of initialize_services
    mock_kb = MagicMock()
    mock_kb_manager_instance = MagicMock()
    mock_kb_manager_instance.load_or_create_kb = AsyncMock(return_value=mock_kb)
    mock_chat_handler_instance = MagicMock()

    # Patch where these objects are *defined* or *imported* in services.py
    with (
        patch("app.services.kb_manager", mock_kb_manager_instance),
        patch(
            "app.services.ChatHandler", return_value=mock_chat_handler_instance
        ) as MockChatHandlerClass,
        patch("app.services.logger") as mock_logger,
    ):

        # Reset the actual app_state before calling
        from app import services

        services.app_state["knowledge_base"] = None
        services.app_state["chat_handler"] = None
        services.app_state["is_kb_loading"] = False

        # Run the initialization function
        await services.initialize_services()

        # Verify calls and state updates
        mock_kb_manager_instance.load_or_create_kb.assert_called_once()
        MockChatHandlerClass.assert_called_once_with(
            knowledge_base=mock_kb, model_name=services.settings.model_name
        )
        mock_logger.info.assert_called()
        assert services.app_state["knowledge_base"] == mock_kb
        assert services.app_state["chat_handler"] == mock_chat_handler_instance
        assert services.app_state["is_kb_loading"] is False


@pytest.mark.asyncio
async def test_initialize_services_failure():
    """Test service initialization failure."""
    # Mock kb_manager to raise an error
    mock_kb_manager_instance = MagicMock()
    mock_kb_manager_instance.load_or_create_kb = AsyncMock(side_effect=Exception("KB Load Error"))

    with (
        patch("app.services.kb_manager", mock_kb_manager_instance),
        patch("app.services.logger") as mock_logger,
    ):

        # Reset state
        from app import services

        services.app_state["knowledge_base"] = None
        services.app_state["chat_handler"] = None
        services.app_state["is_kb_loading"] = False

        await services.initialize_services()

        # Verify error logging and state reset
        mock_logger.error.assert_called()
        assert services.app_state["knowledge_base"] is None
        assert services.app_state["chat_handler"] is None
        assert services.app_state["is_kb_loading"] is False


@pytest.mark.asyncio
async def test_get_current_chat_handler_unavailable():
    """Test the dependency when services are not ready."""
    from app import services

    # Simulate state where loading failed or hasn't finished
    services.app_state["chat_handler"] = None
    services.app_state["is_kb_loading"] = False  # Simulate loading failed

    with pytest.raises(HTTPException) as excinfo:
        await services.get_current_chat_handler()

    assert excinfo.value.status_code == 503
    assert "failed to initialize" in excinfo.value.detail

    # Simulate state where loading is in progress
    services.app_state["is_kb_loading"] = True
    with pytest.raises(HTTPException) as excinfo:
        await services.get_current_chat_handler()

    assert excinfo.value.status_code == 503
    assert "initializing" in excinfo.value.detail


# Testing FastAPI lifespan is complex; focusing on the initialize_services function covers the core logic.
# Testing the lifespan interaction directly often requires more advanced fixtures or async test clients.

# Remove old lifespan test that patched app.main directly
# @pytest.mark.asyncio
# async def test_app_lifespan(): ...
