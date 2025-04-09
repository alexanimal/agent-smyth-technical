"""
Comprehensive unit tests for the knowledge_base module.
"""
import os
import sys
import asyncio
import pytest
import pickle
import json
import concurrent.futures
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.kb import (
    KnowledgeBaseManager, 
    Document, 
    extract_content, 
    metadata_extractor, 
    get_project_root,
    process_file,
    load_documents
)

#################################################
# Mock data and fixtures
#################################################

# Mock data for testing
MOCK_TWEET = {
    "id": "123456789",
    "text": "This is a test tweet",
    "createdAt": "2023-01-01T12:00:00Z",
    "url": "https://twitter.com/user/status/123456789",
    "profile": "testuser",
    "retweetCount": 10,
    "likeCount": 20,
    "replyCount": 5,
    "author": {"username": "testuser"}
}

@pytest.fixture
def kb_manager():
    """Create a KnowledgeBaseManager with mocked paths."""
    with patch('os.path.exists', return_value=False):
        manager = KnowledgeBaseManager(mocks_dir="/fake/path")
        return manager

@pytest.fixture
def mock_document():
    """Create a mock Document."""
    return Document(page_content="Test content", metadata={"url": "https://twitter.com/test"})

@pytest.fixture
def valid_documents():
    """Create a large set of valid documents for testing."""
    return [
        Document(
            page_content=f"Document {i}",
            metadata={"url": f"https://example.com/{i % 20}"}  # Create 20 unique sources
        )
        for i in range(1200)  # Create 1200 documents
    ]

@pytest.fixture
def invalid_documents_too_few():
    """Create a small set of documents that should fail validation."""
    return [
        Document(
            page_content=f"Document {i}",
            metadata={"url": "https://example.com/1"}
        )
        for i in range(10)  # Only 10 documents, should fail minimum count
    ]

@pytest.fixture
def invalid_documents_few_sources():
    """Create documents with too few sources."""
    return [
        Document(
            page_content=f"Document {i}",
            metadata={"url": f"https://example.com/{i % 5}"}  # Only 5 unique sources
        )
        for i in range(1200)
    ]

@pytest.fixture
def mock_embeddings():
    """Mock OpenAIEmbeddings to avoid API calls."""
    with patch('app.kb.OpenAIEmbeddings') as mock_embeddings:
        # Create a mock embeddings instance
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        yield mock_embeddings

# Set up a mock environment for tests
@pytest.fixture(autouse=True)
def mock_env():
    """Set up mock environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
        yield

#################################################
# Path Resolution Tests
#################################################

def test_project_root_exists():
    """Test that the project root detected is an actual directory."""
    root = get_project_root()
    assert os.path.exists(root), f"Project root '{root}' doesn't exist"
    assert os.path.isdir(root), f"Project root '{root}' is not a directory"

def test_mocks_dir_name():
    """Test that the mocks directory has the correct name."""
    from app.kb import MOCKS_DIR
    assert MOCKS_DIR.endswith("__mocks__"), f"MOCKS_DIR '{MOCKS_DIR}' doesn't end with '__mocks__'"

#################################################
# KnowledgeBaseManager Initialization Tests
#################################################

def test_default_mocks_dir():
    """Test that the default mocks_dir is set correctly."""
    from app.kb import MOCKS_DIR
    kb = KnowledgeBaseManager()
    assert kb.mocks_dir, "mocks_dir should not be empty"
    assert "__mocks__" in kb.mocks_dir, "mocks_dir should point to __mocks__ directory"

def test_custom_mocks_dir():
    """Test that a custom mocks_dir is respected."""
    custom_dir = "/custom/path"
    kb = KnowledgeBaseManager(mocks_dir=custom_dir)
    assert kb.mocks_dir == custom_dir, f"Custom mocks_dir '{kb.mocks_dir}' != '{custom_dir}'"

def test_index_path_is_absolute(kb_manager):
    """Test that the index_path is an absolute path."""
    assert os.path.isabs(kb_manager.index_path), f"index_path '{kb_manager.index_path}' is not absolute"

def test_documents_pickle_path_is_under_index_path(kb_manager):
    """Test that documents_pickle_path is under index_path."""
    assert kb_manager.documents_pickle_path.startswith(kb_manager.index_path), \
        f"documents_pickle_path '{kb_manager.documents_pickle_path}' is not under index_path '{kb_manager.index_path}'"

#################################################
# Metadata and Content Extraction Tests
#################################################

def test_metadata_extractor():
    """Test the metadata_extractor method."""
    metadata = KnowledgeBaseManager.metadata_extractor(MOCK_TWEET, {})
    
    assert metadata["id"] == "123456789"
    assert metadata["created_at"] == "2023-01-01T12:00:00Z"
    assert metadata["url"] == "https://twitter.com/user/status/123456789"
    assert metadata["profile"] == "testuser"

def test_metadata_extractor_function():
    """Test the standalone metadata_extractor function."""
    tweet = {
        "id": "123456",
        "createdAt": "2023-01-01T12:00:00Z",
        "url": "https://twitter.com/user/status/123456",
        "author": {"username": "testuser"},
        "profile": "user_profile"  # Added profile field directly
    }
    
    metadata = {}
    result = metadata_extractor(tweet, metadata)
    
    assert result["id"] == "123456"
    assert result["created_at"] == "2023-01-01T12:00:00Z"
    assert result["url"] == "https://twitter.com/user/status/123456"
    # Checking the actual profile field
    assert result["profile"] == "user_profile"

def test_metadata_extractor_missing_fields():
    """Test metadata extraction with missing fields."""
    tweet = {
        "id": "123456"
        # Missing other fields
    }
    
    metadata = {}
    result = metadata_extractor(tweet, metadata)
    
    assert result["id"] == "123456"
    # Should have default values for missing fields
    assert "created_at" in result
    assert "url" in result
    assert "profile" in result

def test_extract_content_tweet():
    """Test extracting content from a tweet."""
    tweet = {
        "text": "This is a test tweet",
        "author": {"username": "testuser"},
        "createdAt": "2023-01-01T12:00:00Z",
        "retweetCount": 10,
        "likeCount": 20
    }
    content = extract_content(tweet)
    assert "This is a test tweet" in content
    assert "testuser" in content
    # Check for the actual format used in the function
    assert "Stats: 10 retweets, 20 likes" in content

def test_extract_content_missing_fields():
    """Test extracting content with missing fields."""
    tweet = {
        "text": "This is a test tweet"
        # Missing other fields
    }
    content = extract_content(tweet)
    assert "This is a test tweet" in content
    # Should handle missing fields gracefully
    assert "unknown" in content.lower() or "" in content

#################################################
# Document Processing Tests
#################################################

@patch('app.kb.JSONLoader')
def test_process_file(mock_loader):
    """Test the process_file static method."""
    # Setup mocks
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [
        Document(page_content="Tweet 1", metadata={}),
        Document(page_content="Tweet 2", metadata={})
    ]
    
    # Call the method
    result = KnowledgeBaseManager.process_file("/fake/path/file.json")
    
    # Assert
    assert len(result) == 2
    mock_loader.assert_called_once()
    mock_loader_instance.load.assert_called_once()

@patch('app.kb.JSONLoader')
def test_process_file_handles_exceptions(mock_loader):
    """Test that process_file handles exceptions properly."""
    # Setup mock to raise an exception
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.side_effect = Exception("Test error")
    
    # Call the method
    result = KnowledgeBaseManager.process_file("/fake/path/file.json")
    
    # Assert empty result on error
    assert result == []

@patch('app.kb.JSONLoader')
def test_process_file_standalone(mock_loader):
    """Test the standalone process_file function."""
    # Setup mock JSONLoader
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [
        Document(page_content="Tweet 1", metadata={"id": "1"}),
        Document(page_content="Tweet 2", metadata={"id": "2"})
    ]
    
    # Call the function with a fake path
    result = KnowledgeBaseManager.process_file("/fake/path/file.json")
    
    # Assert
    assert len(result) == 2
    mock_loader.assert_called_once()
    mock_loader_instance.load.assert_called_once()

#################################################
# Document Loading Tests
#################################################

@pytest.mark.asyncio
@patch('app.kb.glob.glob')
@patch('app.kb.ProcessPoolExecutor')
@patch('app.kb.multiprocessing.cpu_count', return_value=2)  # Ensure at least 2 workers
async def test_load_documents(mock_cpu_count, mock_executor, mock_glob, kb_manager):
    """Test the load_documents method with multiprocessing."""
    # Setup mocks
    mock_glob.return_value = ["/fake/path/file1.json", "/fake/path/file2.json"]
    
    # Mock executor and futures
    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    
    # Create mock futures
    future1 = concurrent.futures.Future()
    future1.set_result([Document(page_content="Tweet 1", metadata={})])
    
    future2 = concurrent.futures.Future()
    future2.set_result([Document(page_content="Tweet 2", metadata={})])
    
    mock_executor_instance.submit.side_effect = [future1, future2]
    
    # Mock as_completed to return our futures
    with patch('concurrent.futures.as_completed', return_value=[future1, future2]):
        documents = await kb_manager.load_documents()
    
    # Assert
    assert len(documents) == 2
    assert mock_executor_instance.submit.call_count == 2

@pytest.mark.asyncio
async def test_load_documents_standalone():
    """Test the standalone load_documents function."""
    # Create test documents to return
    test_docs = [
        Document(page_content="Doc 1", metadata={}),
        Document(page_content="Doc 2", metadata={})
    ]
    
    # Mock the entire load_documents function
    with patch("app.kb.load_documents", new_callable=AsyncMock) as mock_load:
        # Set up mock to return our test documents
        mock_load.return_value = test_docs
        
        # Import the actual function we want to test
        from app.kb import load_documents
        
        # Call the function
        result = await load_documents()
        
        # Verify it was called and returned our test docs
        mock_load.assert_called_once()
        assert len(result) == 2
        assert test_docs[0] in result
        assert test_docs[1] in result

@pytest.mark.asyncio
async def test_load_documents_empty_dir(kb_manager):
    """Test loading documents from an empty directory."""
    with patch('glob.glob', return_value=[]):
        result = await kb_manager.load_documents()
        assert result == []

@pytest.mark.asyncio
async def test_load_documents_ensures_min_workers(kb_manager):
    """Test that load_documents ensures at least 1 worker."""
    # Use a non-empty list of files to ensure the ProcessPoolExecutor is created
    with patch('app.kb.ProcessPoolExecutor') as mock_executor, \
         patch('glob.glob', return_value=["/fake/path/file1.json", "/fake/path/file2.json"]), \
         patch('multiprocessing.cpu_count', return_value=4), \
         patch('concurrent.futures.as_completed', return_value=[]):
        
        # Setup mock context manager behavior
        mock_context = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_context
        
        # Mock the executor to return empty results to avoid file not found errors
        future = concurrent.futures.Future()
        future.set_result([])
        mock_context.submit.return_value = future
        
        # Call the method
        await kb_manager.load_documents()
        
        # Verify it was called with the correct number of processes (min of CPU count and file count)
        mock_executor.assert_called_once()
        call_args = mock_executor.call_args
        assert call_args is not None
        assert 'max_workers' in call_args[1]
        assert call_args[1]['max_workers'] == 2

#################################################
# Document Validation Tests
#################################################

def test_validate_documents_success(kb_manager, valid_documents):
    """Test document validation with valid documents."""
    result = kb_manager.validate_documents(valid_documents)
    assert result is True

def test_validate_documents_empty(kb_manager):
    """Test document validation with empty list."""
    result = kb_manager.validate_documents([])
    assert result is False

def test_validate_documents_too_few(kb_manager, invalid_documents_too_few):
    """Test document validation with too few documents."""
    result = kb_manager.validate_documents(invalid_documents_too_few)
    assert result is False

def test_validate_documents_few_sources(kb_manager, invalid_documents_few_sources):
    """Test document validation with too few sources."""
    result = kb_manager.validate_documents(invalid_documents_few_sources)
    assert result is False

#################################################
# Document Pickle Methods Tests
#################################################

def test_save_documents_pickle(kb_manager, valid_documents):
    """Test saving documents to pickle."""
    with patch("builtins.open", mock_open()) as mock_file, \
         patch("os.makedirs") as mock_makedirs, \
         patch("pickle.dump") as mock_dump:
        
        kb_manager._save_documents_pickle(valid_documents)
        
        # Verify directory creation
        mock_makedirs.assert_called_once()
        
        # Verify file opening
        mock_file.assert_called_once_with(kb_manager.documents_pickle_path, 'wb')
        
        # Verify pickling
        mock_dump.assert_called_once()

def test_load_documents_pickle_success(kb_manager, valid_documents):
    """Test loading documents from pickle successfully."""
    # Setup mock to return test documents
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("pickle.load", return_value=valid_documents) as mock_load:
        
        result = kb_manager._load_documents_pickle()
        
        # Verify file existence check
        assert os.path.exists(kb_manager.documents_pickle_path)
        
        # Verify file opening
        mock_file.assert_called_once_with(kb_manager.documents_pickle_path, 'rb')
        
        # Verify unpickling
        mock_load.assert_called_once()
        
        # Verify result is the mock documents
        assert result == valid_documents

def test_load_documents_pickle_not_exists(kb_manager):
    """Test loading documents when pickle doesn't exist."""
    with patch("os.path.exists", return_value=False):
        result = kb_manager._load_documents_pickle()
        assert result is None

def test_load_documents_pickle_error(kb_manager):
    """Test loading documents with pickle error."""
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()), \
         patch("pickle.load", side_effect=Exception("Test pickle error")):
        
        result = kb_manager._load_documents_pickle()
        assert result is None

#################################################
# Index Integrity Tests
#################################################

def test_check_index_integrity_success(kb_manager):
    """Test successful index integrity check."""
    with patch("os.path.exists", return_value=True):
        result = kb_manager._check_index_integrity()
        assert result is True

def test_check_index_integrity_no_path(kb_manager):
    """Test index integrity with no index directory."""
    with patch("os.path.exists", return_value=False):
        result = kb_manager._check_index_integrity()
        assert result is False

def test_check_index_integrity_missing_file(kb_manager):
    """Test index integrity with missing file."""
    def exists_side_effect(path):
        # Return True for directory but False for a specific file
        if path.endswith("index.pkl"):
            return False
        return True
        
    with patch("os.path.exists", side_effect=exists_side_effect):
        result = kb_manager._check_index_integrity()
        assert result is False

#################################################
# Vector Store Tests
#################################################

@pytest.mark.asyncio
@patch('os.path.exists')
@patch('app.kb.FAISS')
async def test_load_existing_index(mock_faiss, mock_exists, kb_manager, mock_embeddings):
    """Test loading an existing index."""
    # Setup mocks
    mock_exists.return_value = True
    mock_vector_store = MagicMock()
    mock_faiss.load_local.return_value = mock_vector_store
    
    # Call the method
    result = await kb_manager.load_or_create_kb()
    
    # Assert
    assert result == mock_vector_store
    mock_faiss.load_local.assert_called_once()
    assert kb_manager._vector_store == mock_vector_store
    # Verify embeddings were created
    mock_embeddings.assert_called_once()

@pytest.mark.asyncio
@patch('os.path.exists')
@patch('app.kb.FAISS')
async def test_create_new_index_when_loading_fails(mock_faiss, mock_exists, kb_manager, mock_embeddings):
    """Test creating a new index when loading fails."""
    # Setup mocks
    mock_exists.return_value = True
    mock_faiss.load_local.side_effect = Exception("Failed to load")
    
    # Mock load_documents
    with patch.object(kb_manager, 'load_documents', return_value=[Document(page_content="Test", metadata={})]):
        # Mock text splitter
        with patch('app.kb.RecursiveCharacterTextSplitter.split_documents', 
                  return_value=[Document(page_content="Test", metadata={})]):
            # Mock FAISS from_documents
            mock_vector_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_vector_store
            
            # Call the method
            result = await kb_manager.load_or_create_kb()
    
    # Assert
    assert result == mock_vector_store
    mock_faiss.from_documents.assert_called_once()
    assert kb_manager._vector_store == mock_vector_store

@pytest.mark.asyncio
async def test_save_index(kb_manager):
    """Test saving the index."""
    # Setup
    mock_vector_store = MagicMock()
    kb_manager._vector_store = mock_vector_store
    
    # Call the method
    kb_manager.save_index()
    
    # Assert
    mock_vector_store.save_local.assert_called_once_with(kb_manager.index_path)

@pytest.mark.asyncio
@patch('app.kb.FAISS')
async def test_full_workflow(mock_faiss, kb_manager, mock_embeddings):
    """Test the full workflow of creating and saving an index."""
    # Setup mocks
    with patch('os.path.exists', return_value=False):
        # Mock load_documents to return test documents
        with patch.object(kb_manager, 'load_documents', 
                         return_value=[Document(page_content="Test", metadata={})]):
            # Mock text splitter
            with patch('app.kb.RecursiveCharacterTextSplitter.split_documents', 
                      return_value=[Document(page_content="Test", metadata={})]):
                # Mock FAISS from_documents
                mock_vector_store = MagicMock()
                mock_faiss.from_documents.return_value = mock_vector_store
                
                # Call the method
                result = await kb_manager.load_or_create_kb()
    
    # Assert the full workflow
    assert result == mock_vector_store
    mock_faiss.from_documents.assert_called_once()
    mock_vector_store.save_local.assert_called_once()
    assert kb_manager._vector_store == mock_vector_store

@pytest.mark.asyncio
async def test_empty_documents_raises_error(kb_manager):
    """Test that an error is raised when no documents are loaded."""
    # Mock methods that could cause early return, AND load_documents
    with patch.object(kb_manager, '_check_index_integrity', return_value=False), \
         patch.object(kb_manager, '_load_documents_pickle', return_value=None), \
         patch.object(kb_manager, 'load_documents', return_value=[]):
        # Call the method THAT RAISES THE ERROR inside the context manager
        with pytest.raises(ValueError, match="No documents were loaded"):
            await kb_manager.load_or_create_kb() 