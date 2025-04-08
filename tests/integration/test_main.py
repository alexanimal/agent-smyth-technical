"""
Integration tests for the FastAPI application in main.py
"""
import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create a patch for the KnowledgeBaseManager before importing the app
with patch('src.main.KnowledgeBaseManager') as mock_kb_manager, \
     patch('src.main.ChatHandler') as mock_chat_handler:
    # Now import the app after patching
    from src.app import app, load_kb_in_background, get_chat_handler

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
        "query_type": "general"
    }


@pytest.fixture
def setup_mocks(mock_knowledge_base, mock_chat_response):
    """Setup all necessary mocks."""
    # Reset the global variables in main.py
    import src.app
    src.app.knowledge_base = mock_knowledge_base
    src.app.is_kb_loading = False
    
    # Create a mock chat handler
    mock_handler = AsyncMock()
    mock_handler.process_query = AsyncMock(return_value=mock_chat_response)
    src.app.chat_handler = mock_handler
    
    return mock_handler


def test_root_endpoint(setup_mocks):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "ready"  # Since we mocked knowledge_base to be loaded


def test_health_check(setup_mocks):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "knowledge_base_loaded": True,
        "is_loading": False
    }


def test_chat_endpoint(setup_mocks):
    """Test the chat endpoint."""
    # Setting up the mock for the dependency
    with patch('src.main.get_chat_handler', return_value=setup_mocks):
        # Make the request
        response = client.post(
            "/chat",
            json={"message": "Test message", "num_results": 3}
        )
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {
            "response": "This is a test response",
            "sources": ["https://example.com/1", "https://example.com/2"],
            "processing_time": 0.5
        }
        
        # Verify the mock was called correctly
        setup_mocks.process_query.assert_called_once_with(
            message="Test message",
            k=3
        )


def test_chat_endpoint_error(setup_mocks):
    """Test error handling in the chat endpoint."""
    # Make the handler raise an exception
    setup_mocks.process_query.side_effect = Exception("Test error")
    
    # Setting up the mock for the dependency
    with patch('src.main.get_chat_handler', return_value=setup_mocks):
        # Make the request
        response = client.post(
            "/chat",
            json={"message": "Test message"}
        )
        
        # Check error response
        assert response.status_code == 500
        assert "error" in response.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_load_kb_in_background_success():
    """Test successful knowledge base loading."""
    # Setup mocks
    mock_kb = AsyncMock()
    mock_kb_manager = MagicMock()
    mock_kb_manager.load_or_create_kb = AsyncMock(return_value=mock_kb)
    
    with patch('src.main.kb_manager', mock_kb_manager), \
         patch('src.main.logger') as mock_logger, \
         patch('src.main.ChatHandler') as mock_chat_handler:
        
        # Reset global variables
        import src.app
        src.app.knowledge_base = None
        src.app.chat_handler = None
        src.app.is_kb_loading = False
        
        # Run the function
        await src.app.load_kb_in_background()
        
        # Verify it was called and global variables updated
        mock_kb_manager.load_or_create_kb.assert_called_once()
        mock_logger.info.assert_called()
        assert src.app.knowledge_base is not None
        assert src.app.chat_handler is not None
        assert src.app.is_kb_loading is False


@pytest.mark.asyncio
async def test_load_kb_in_background_failure():
    """Test knowledge base loading failure."""
    # Setup mocks to simulate failure
    mock_kb_manager = MagicMock()
    mock_kb_manager.load_or_create_kb = AsyncMock(side_effect=Exception("Test error"))
    
    with patch('src.main.kb_manager', mock_kb_manager), \
         patch('src.main.logger') as mock_logger:
        
        # Reset global variables
        import src.app
        src.app.knowledge_base = None
        src.app.chat_handler = None
        src.app.is_kb_loading = False
        
        # Run the function
        await src.app.load_kb_in_background()
        
        # Verify error handling
        mock_logger.error.assert_called()
        assert src.app.knowledge_base is None
        assert src.app.chat_handler is None
        assert src.app.is_kb_loading is False


@pytest.mark.asyncio
async def test_get_chat_handler_service_unavailable():
    """Test the get_chat_handler dependency when KB is not loaded."""
    # Reset global var to simulate KB not loaded
    import src.app
    src.app.knowledge_base = None
    
    with pytest.raises(Exception) as excinfo:
        await src.app.get_chat_handler()
    
    assert "not yet initialized" in str(excinfo.value)


# Test FastAPI lifespan is more challenging, but could be done with AsyncMock
@pytest.mark.asyncio
async def test_app_lifespan():
    """Test the app lifespan."""
    with patch('src.main.load_kb_in_background') as mock_load, \
         patch('asyncio.create_task') as mock_create_task:
        
        # Create a simple mock context manager that simulates the lifespan event
        import src.app
        lifespan_cm = src.app.lifespan(app)
        
        # Trigger the startup
        await lifespan_cm.__aenter__()
        
        # Verify background task was created
        mock_create_task.assert_called_once()
        
        # Trigger the shutdown
        await lifespan_cm.__aexit__(None, None, None) 