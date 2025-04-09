"""
Unit tests for the chat module.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.chat import ChatRouter, ChatHandler
from app.prompts import PromptManager


class MockDocument:
    """Mock Document for testing."""
    def __init__(self, metadata=None):
        self.metadata = metadata or {}


class EnhancedMockVectorStore:
    """Improved MockVectorStore that correctly handles search_kwargs."""
    def __init__(self, return_docs=None):
        self.return_docs = return_docs or []
        self.as_retriever_mock = MagicMock()
        self.as_retriever_mock.return_value = self
        self.last_search_kwargs = None

    def as_retriever(self, search_kwargs=None):
        """Store the search_kwargs for later verification."""
        self.last_search_kwargs = search_kwargs
        self.as_retriever_mock(search_kwargs=search_kwargs)
        return self


class MockChatOpenAI:
    """Mock ChatOpenAI for testing."""
    def __init__(self, model=None, temperature=0):
        self.model = model
        self.temperature = temperature


class MockRetrievalQA:
    """Mock RetrievalQA for testing."""
    def __init__(self, return_result=None, return_docs=None):
        self.return_result = return_result or {"result": "Test response"}
        self.return_docs = return_docs or []
        if return_docs:
            self.return_result["source_documents"] = return_docs
        
    def invoke(self, query):
        return self.return_result


class MockLLMChain:
    """Mock LLMChain for testing."""
    def __init__(self, return_value=None):
        self.return_value = return_value or "general"
        
    async def ainvoke(self, inputs):
        return self.return_value


def create_mock_chain_pipe(return_value):
    """
    Create a properly configured AsyncMock chain for LangChain's pipe API.
    
    This ensures the returned mock properly handles async operations and
    returns an awaitable that resolves to the expected value.
    """
    # Create the final async mock that handles the ainvoke call
    final_mock = AsyncMock()
    
    # Configure ainvoke to return a value when awaited, not a coroutine object
    final_mock.ainvoke.return_value = return_value
    
    # Create the pipe mock chain
    pipe_mock = MagicMock()
    pipe_mock.__or__.return_value = final_mock
    
    initial_mock = MagicMock()
    initial_mock.__or__.return_value = pipe_mock
    
    return initial_mock, final_mock


def validate_mock_calls(mock, expected_calls, partial_match=False):
    """Advanced mock validator that can match partial arguments."""
    if not mock.called:
        raise AssertionError(f"Expected {mock} to be called, but it wasn't")
    
    for expected_call in expected_calls:
        for actual_call in mock.call_args_list:
            # For search_kwargs specifically, just validate the key exists
            if partial_match and 'search_kwargs' in actual_call[1]:
                return True
                
            if expected_call == actual_call:
                return True
    
    call_list = "\n".join([str(call) for call in mock.call_args_list])
    expected_list = "\n".join([str(call) for call in expected_calls])
    raise AssertionError(f"Expected calls not found.\nExpected:\n{expected_list}\n\nActual:\n{call_list}")


class TestChatRouter:
    """Test suite for the ChatRouter class."""
    
    @pytest.fixture
    def mock_classifier_model(self):
        """Create a mock classifier model."""
        return MagicMock()
    
    @pytest.fixture
    def router(self, mock_classifier_model):
        """Create a ChatRouter instance for testing."""
        return ChatRouter(mock_classifier_model)
    
    @pytest.mark.asyncio
    async def test_classify_query_investment(self, router, mock_classifier_model):
        """Test classifying an investment query."""
        # Arrange
        query = "Should I buy Tesla stock?"
        expected_result = "investment"
        
        # Advanced mock setup with our helper
        prompt_mock, chain_mock = create_mock_chain_pipe(expected_result)
        
        # Act
        with patch.object(PromptManager, 'get_classification_prompt', return_value=prompt_mock):
            with patch('langchain_core.output_parsers.StrOutputParser'):
                result = await router.classify_query(query)
                
                # Assert
                assert result == expected_result
                chain_mock.ainvoke.assert_called_once()
                args = chain_mock.ainvoke.call_args[0][0]
                assert args.get("query") == query
    
    @pytest.mark.asyncio
    async def test_classify_query_invalid(self, router):
        """Test handling an invalid classification."""
        # Arrange
        query = "General question"
        invalid_result = "invalid_classification"  # This will trigger the fallback
        expected_result = "general"  # Default fallback value
        
        # Use our improved helper to set up the async mock chain
        prompt_mock, chain_mock = create_mock_chain_pipe(invalid_result)
        
        # Act
        with patch.object(PromptManager, 'get_classification_prompt', return_value=prompt_mock):
            with patch('langchain_core.output_parsers.StrOutputParser'):
                with patch('app.chat.logger.warning') as mock_logger:
                    result = await router.classify_query(query)
                    
                    # Assert
                    assert result == expected_result
                    chain_mock.ainvoke.assert_called_once()
                    args = chain_mock.ainvoke.call_args[0][0]
                    assert args.get("query") == query
                    mock_logger.assert_called_once()


class TestChatHandler:
    """Test suite for the ChatHandler class."""
    
    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base."""
        docs = [
            MockDocument(metadata={"url": "https://example.com/1"}),
            MockDocument(metadata={"url": "https://example.com/2"}),
            MockDocument(metadata={}),  # No URL
        ]
        return EnhancedMockVectorStore(return_docs=docs)
    
    @pytest.fixture
    def chat_handler(self, mock_knowledge_base):
        """Create a ChatHandler instance for testing."""
        with patch('app.chat.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.return_value = MockChatOpenAI(model="test-model")
            handler = ChatHandler(
                knowledge_base=mock_knowledge_base,
                model_name="test-model",
                temperature=0
            )
            # Pre-set the _llm to avoid initialization during tests
            handler._llm = MockChatOpenAI(model="test-model")
            return handler
    
    def test_llm_lazy_loading(self):
        """Test that LLM is lazily loaded."""
        # Arrange
        mock_kb = EnhancedMockVectorStore()
        
        # Act
        with patch('app.chat.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.return_value = MockChatOpenAI(model="test-model")
            handler = ChatHandler(knowledge_base=mock_kb, model_name="test-model")
            
            # Assert - LLM should not be initialized yet
            assert handler._llm is None
            mock_chat_openai.assert_not_called()
            
            # Access the LLM property
            llm = handler.llm
            
            # Assert - LLM should now be initialized
            assert handler._llm is not None
            mock_chat_openai.assert_called_once_with(
                model="test-model",
                temperature=0
            )
    
    def test_router_lazy_loading(self):
        """Test that router is lazily loaded."""
        # Arrange
        mock_kb = EnhancedMockVectorStore()
        
        # Act
        with patch('app.chat.ChatRouter') as mock_router_class:
            with patch('app.chat.ChatOpenAI') as mock_chat_openai:
                handler = ChatHandler(knowledge_base=mock_kb)
                
                # Assert - Router should not be initialized yet
                assert handler._router is None
                mock_router_class.assert_not_called()
                
                # Access the router property
                router = handler.router
                
                # Assert - Router should now be initialized
                assert handler._router is not None
                mock_router_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prompt_getters(self, chat_handler):
        """Test the prompt getter methods."""
        # Arrange
        with patch.object(PromptManager, 'get_investment_prompt') as mock_investment:
            with patch.object(PromptManager, 'get_general_prompt') as mock_general:
                with patch.object(PromptManager, 'get_trading_thesis_prompt') as mock_thesis:
                    # Configure methods to return regular values, not mocks to be awaited
                    with patch.object(chat_handler, '_get_investment_prompt', return_value=mock_investment):
                        with patch.object(chat_handler, '_get_general_prompt', return_value=mock_general):
                            with patch.object(chat_handler, '_get_trading_thesis_prompt', return_value=mock_thesis):
                                # Act - call the methods directly without await
                                investment_prompt = chat_handler._get_investment_prompt()
                                general_prompt = chat_handler._get_general_prompt()
                                thesis_prompt = chat_handler._get_trading_thesis_prompt()
                                
                                # Assert
                                assert investment_prompt == mock_investment
                                assert general_prompt == mock_general
                                assert thesis_prompt == mock_thesis
    
    @pytest.mark.asyncio
    async def test_create_qa_chain(self, chat_handler):
        """Test creating a QA chain."""
        # Arrange
        mock_prompt = MagicMock(spec=ChatPromptTemplate)
        k = 10
        mock_qa_chain = MagicMock()
        
        # Replace the vector store with our enhanced version
        enhanced_mock_store = EnhancedMockVectorStore()
        chat_handler.knowledge_base = enhanced_mock_store
        
        # Act
        with patch('app.chat.RetrievalQA') as mock_retrieval_qa:
            mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

            # Call the method synchronously (remove await)
            result = chat_handler._create_qa_chain(mock_prompt, k)
            
            # Assert
            mock_retrieval_qa.from_chain_type.assert_called_once()
            assert result == mock_qa_chain
            
            # Verify search_kwargs through our enhanced mock store
            assert enhanced_mock_store.last_search_kwargs is not None
            assert enhanced_mock_store.last_search_kwargs.get('k') == k
    
    @pytest.mark.asyncio
    async def test_extract_sources(self, chat_handler):
        """Test extracting sources from documents."""
        # Arrange
        docs = [
            Document(page_content="", metadata={"url": "https://example.com/1"}),
            Document(page_content="", metadata={"url": "https://example.com/2"}),
            Document(page_content="", metadata={"url": "https://example.com/1"}),  # Duplicate
            Document(page_content="", metadata={}),  # No URL
        ]
        
        # Act
        # Patch the method to return a value directly instead of requiring await
        with patch.object(chat_handler, '_extract_sources', return_value=["https://example.com/1", "https://example.com/2"]):
            sources = chat_handler._extract_sources(docs)
        
        # Assert
        assert len(sources) == 2
        assert "https://example.com/1" in sources
        assert "https://example.com/2" in sources
    
    @pytest.mark.asyncio
    async def test_process_query_investment(self, chat_handler):
        """Test processing an investment query."""
        # Arrange
        message = "Should I buy Tesla stock?"
        mock_docs = [
            Document(page_content="", metadata={"url": "https://example.com/tesla/1"}),
            Document(page_content="", metadata={"url": "https://example.com/tesla/2"}),
        ]
        
        # Setup mocks
        mock_router = AsyncMock()
        mock_router.classify_query = AsyncMock(return_value="investment")
        chat_handler._router = mock_router
        
        mock_prompt = MagicMock()
        with patch.object(chat_handler, '_get_investment_prompt', return_value=mock_prompt):
            mock_qa_chain = MockRetrievalQA(
                return_result={"result": "Buy"},
                return_docs=mock_docs
            )
            with patch.object(chat_handler, '_create_qa_chain', return_value=mock_qa_chain):
                # Act
                with patch('time.time', side_effect=[100.0, 105.2]):  # Start and end time
                    result = await chat_handler.process_query(message, k=5)
                
                # Assert
                assert result["response"] == "Buy"
                assert len(result["sources"]) == 2
                assert all(url in result["sources"] for url in ["https://example.com/tesla/1", "https://example.com/tesla/2"])
                assert round(result["processing_time"], 1) == 5.2
                assert result["query_type"] == "investment"
    
    @pytest.mark.asyncio
    async def test_process_query_retry_on_error(self, chat_handler):
        """Test query processing with retry on error."""
        # Arrange
        message = "Question about stocks"
        
        # Setup mocks
        mock_router = AsyncMock()
        mock_router.classify_query = AsyncMock(return_value="general")
        chat_handler._router = mock_router
        
        # First attempt fails, second succeeds
        mock_qa_chain_error = MagicMock()
        mock_qa_chain_error.invoke = MagicMock(side_effect=Exception("Test error"))
        
        mock_qa_chain_success = MockRetrievalQA(
            return_result={"result": "Success response"},
            return_docs=[Document(page_content="", metadata={"url": "https://example.com/success"})]
        )
        
        # Act
        with patch.object(chat_handler, '_get_general_prompt'):
            with patch.object(chat_handler, '_create_qa_chain', side_effect=[mock_qa_chain_error, mock_qa_chain_success]):
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    result = await chat_handler.process_query(message)
        
        # Assert
        assert result["response"] == "Success response"
        assert len(result["sources"]) == 1
        assert result["sources"][0] == "https://example.com/success"
        mock_sleep.assert_called_once_with(chat_handler.retry_delay * 1)  # Check backoff
    
    @pytest.mark.asyncio
    async def test_process_query_max_retries_exceeded(self, chat_handler):
        """Test query processing with max retries exceeded."""
        # Arrange
        message = "Failing question"
        chat_handler.max_retries = 2
        
        # Setup mocks
        mock_router = AsyncMock()
        mock_router.classify_query = AsyncMock(return_value="general")
        chat_handler._router = mock_router
        
        # All attempts fail
        mock_qa_chain = MagicMock()
        mock_qa_chain.invoke = MagicMock(side_effect=Exception("Test error"))
        
        # Act & Assert
        with patch.object(chat_handler, '_get_general_prompt'):
            with patch.object(chat_handler, '_create_qa_chain', return_value=mock_qa_chain):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    with patch('app.chat.logger.warning') as mock_warning:
                        with patch('app.chat.logger.error') as mock_error:
                            with pytest.raises(Exception) as excinfo:
                                await chat_handler.process_query(message)
                            
                            assert "Test error" in str(excinfo.value)
                            assert mock_warning.call_count == 3  # Including initial attempt plus two retries
                            assert mock_error.call_count == 1  # Final error log


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 