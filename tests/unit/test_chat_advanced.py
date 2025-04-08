"""
Advanced unit tests for the chat module focusing on edge cases and uncovered code paths.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.chat import ChatRouter, ChatHandler

@pytest.mark.asyncio
async def test_process_query_with_max_retries():
    """Test the process_query method with maximum retries and backoff."""
    # Create mocks
    mock_kb = MagicMock()
    mock_router = AsyncMock()
    mock_router.classify_query = AsyncMock(return_value="trading_thesis")
    
    # Setup handler with router directly
    handler = ChatHandler(knowledge_base=mock_kb, max_retries=2, retry_delay=0.1)
    handler._router = mock_router
    
    # Mock the methods
    with patch.object(handler, '_get_trading_thesis_prompt') as mock_get_prompt, \
         patch.object(handler, '_create_qa_chain') as mock_create_chain, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep, \
         patch('src.chat.logger.warning') as mock_warning, \
         patch('src.chat.logger.error') as mock_error:
        
        # Set up chain to always fail
        mock_get_prompt.return_value = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(side_effect=Exception("Test error"))
        mock_create_chain.return_value = mock_chain
        
        # Invoke with error
        with pytest.raises(Exception) as excinfo:
            await handler.process_query("Test query")
        
        # Verify error message
        assert "Test error" in str(excinfo.value)
        
        # Verify retry logic
        assert mock_warning.call_count == 3  # Initial attempt + 2 retries
        assert mock_error.call_count == 1  # Final error log
        
        # Verify increasing backoff
        mock_sleep.assert_has_calls([
            call(0.1),  # First retry with delay * 1
            call(0.2),  # Second retry with delay * 2
        ])
        
        # Verify trading_thesis specific behavior
        assert mock_create_chain.call_count == 3  # Called for each attempt
        # Verify k param is increased for trading_thesis
        k_values = [call_args[0][1] for call_args in mock_create_chain.call_args_list]
        assert all(k >= 10 for k in k_values)  # Should be at least 10 for trading_thesis

@pytest.mark.asyncio
async def test_process_query_trading_thesis():
    """Test the process_query method specifically for trading thesis queries."""
    # Create mocks
    mock_kb = MagicMock()
    mock_router = AsyncMock()
    mock_router.classify_query = AsyncMock(return_value="trading_thesis")
    
    # Setup handler with router directly
    handler = ChatHandler(knowledge_base=mock_kb)
    handler._router = mock_router
    
    # Test message
    message = "Convert this PM note into a trading thesis"
    
    # Mock response data
    mock_result = {
        "result": "This is a trading thesis",
        "source_documents": []
    }
    
    # Mock the methods - Need 4 time values - 2 for each process_query call
    with patch.object(handler, '_get_trading_thesis_prompt') as mock_get_prompt, \
         patch.object(handler, '_create_qa_chain') as mock_create_chain, \
         patch('time.time', side_effect=[100.0, 101.0, 102.0, 103.0]), \
         patch('src.chat.logger') as mock_logger:  # Mock logger to prevent issues
    
        # Set up mock chain
        mock_get_prompt.return_value = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=mock_result)
        mock_create_chain.return_value = mock_chain
        
        # Standard k value
        k = 5
        
        # Process query
        result = await handler.process_query(message, k=k)
        
        # Verify response
        assert result["response"] == "This is a trading thesis"
        assert result["processing_time"] == 1.0
        assert result["query_type"] == "trading_thesis"
        
        # Verify that k was increased for trading_thesis
        # The specific line we're testing: k = max(k, 10)
        mock_create_chain.assert_called_once()
        actual_k = mock_create_chain.call_args[0][1]
        assert actual_k == 10  # Should be adjusted to 10 minimum
        
        # Test with higher k
        mock_create_chain.reset_mock()
        result = await handler.process_query(message, k=15)
        # Should keep the higher k value
        assert mock_create_chain.call_args[0][1] == 15
        assert result["processing_time"] == 1.0  # Second start/end time difference

@pytest.mark.asyncio
async def test_process_query_default_to_general():
    """Test that process_query defaults to general when classification is invalid."""
    # Create mocks
    mock_kb = MagicMock()
    mock_router = AsyncMock()
    # Return an invalid classification
    mock_router.classify_query = AsyncMock(return_value="invalid_type")
    
    # Setup handler with router directly
    handler = ChatHandler(knowledge_base=mock_kb)
    handler._router = mock_router
    
    # Test message
    message = "Random question"
    
    # Mock response data
    mock_result = {
        "result": "General response",
        "source_documents": []
    }
    
    # Mock the methods
    with patch.object(handler, '_get_general_prompt') as mock_get_prompt, \
         patch.object(handler, '_create_qa_chain') as mock_create_chain, \
         patch('time.time', side_effect=[100.0, 101.0]), \
         patch('src.chat.logger.warning') as mock_warning:
    
        # Set up mock chain
        mock_get_prompt.return_value = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=mock_result)
        mock_create_chain.return_value = mock_chain
        
        # Process query
        result = await handler.process_query(message)
        
        # Should default to general prompt when classification is invalid
        mock_get_prompt.assert_called_once()
        assert result["query_type"] == "invalid_type"  # Should still keep original classification
        
        # No warning should be logged since general is the default anyway
        mock_warning.assert_not_called() 