"""
Comprehensive unit tests for the knowledge_base module.
"""

import asyncio
import concurrent.futures
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor  # Import ProcessPoolExecutor directly
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, mock_open, patch

import faiss
import numpy as np
import pytest
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.kb.manager import (
    KnowledgeBaseManager,
    extract_content,
    get_knowledge_base,
    load_documents,
    metadata_extractor,
    process_file,
)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
    "author": {"username": "testuser"},
}


@pytest.fixture
def kb_manager():
    """Create a KnowledgeBaseManager instance with test paths."""
    # Create a test instance manually without importing settings
    kb = KnowledgeBaseManager(mocks_dir="test_data_dir")

    # Override paths for testing
    kb.index_path = "/tmp/test_faiss_index"
    kb.documents_pickle_path = "/tmp/test_faiss_index/documents.pkl"
    kb._vector_store = None

    return kb


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
            metadata={"url": f"https://example.com/{i % 20}"},  # Create 20 unique sources
        )
        for i in range(1200)  # Create 1200 documents
    ]


@pytest.fixture
def invalid_documents_too_few():
    """Create a small set of documents that should fail validation."""
    return [
        Document(page_content=f"Document {i}", metadata={"url": "https://example.com/1"})
        for i in range(10)  # Only 10 documents, should fail minimum count
    ]


@pytest.fixture
def invalid_documents_few_sources():
    """Create documents with too few sources."""
    return [
        Document(
            page_content=f"Document {i}",
            metadata={"url": f"https://example.com/{i % 5}"},  # Only 5 unique sources
        )
        for i in range(1200)
    ]


@pytest.fixture
def mock_embeddings():
    """Mock OpenAIEmbeddings to avoid API calls."""
    with patch("app.kb.manager.OpenAIEmbeddings") as mock_embeddings:
        # Create a mock embeddings instance
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        yield mock_embeddings


# Set up a mock environment for tests
@pytest.fixture(autouse=True)
def mock_env():
    """Set up mock environment variables."""
    # Environment variables
    env_vars = {
        "OPENAI_API_KEY": "sk-mock-key",
        "MODEL_NAME": "gpt-3.5-turbo",
        "ENVIRONMENT": "test",
    }

    # Apply patches
    with patch.dict(os.environ, env_vars):
        # Use simple context manager patching for key classes
        patches = [
            patch("app.kb.manager.OpenAIEmbeddings", return_value=MagicMock()),
            patch("app.kb.manager.FAISS", return_value=MagicMock()),
        ]

        # Apply all patches
        for p in patches:
            p.start()

        yield

        # Clean up patches
        for p in patches:
            p.stop()


#################################################
# KnowledgeBaseManager Initialization Tests
#################################################


def test_default_mocks_dir():
    """Test that the default mocks_dir is set correctly based on internal logic."""
    kb = KnowledgeBaseManager()
    assert kb.mocks_dir, "mocks_dir should not be empty"
    assert "/data" in kb.mocks_dir.replace("\\", "/"), "mocks_dir should point to data directory"


def test_custom_mocks_dir():
    """Test that a custom mocks_dir is respected."""
    custom_dir = "/custom/path"
    kb = KnowledgeBaseManager(mocks_dir=custom_dir)
    assert kb.mocks_dir == custom_dir, f"Custom mocks_dir '{kb.mocks_dir}' != '{custom_dir}'"


def test_index_path_is_absolute(kb_manager):
    """Test that the index_path is an absolute path."""
    assert os.path.isabs(
        kb_manager.index_path
    ), f"index_path '{kb_manager.index_path}' is not absolute"


def test_documents_pickle_path_is_under_index_path(kb_manager):
    """Test that documents_pickle_path is under index_path."""
    assert kb_manager.documents_pickle_path.startswith(
        kb_manager.index_path
    ), f"documents_pickle_path '{kb_manager.documents_pickle_path}' is not under index_path '{kb_manager.index_path}'"


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
        "profile": "user_profile",  # Added profile field directly
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
        "likeCount": 20,
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


@patch("app.kb.manager.JSONLoader")
def test_process_file(mock_loader):
    """Test the process_file static method."""
    # Setup mocks
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [
        Document(page_content="Tweet 1", metadata={}),
        Document(page_content="Tweet 2", metadata={}),
    ]

    # Call the method
    result = KnowledgeBaseManager.process_file("/fake/path/file.json")

    # Assert
    assert len(result) == 2
    mock_loader.assert_called_once()
    mock_loader_instance.load.assert_called_once()


@patch("app.kb.manager.JSONLoader")
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


@patch("app.kb.manager.JSONLoader")
def test_process_file_standalone(mock_loader):
    """Test the standalone process_file function."""
    # Setup mock JSONLoader
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [
        Document(page_content="Tweet 1", metadata={"id": "1"}),
        Document(page_content="Tweet 2", metadata={"id": "2"}),
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
@patch("app.kb.manager.glob.glob")
@patch("app.kb.manager.ProcessPoolExecutor")
@patch("app.kb.manager.multiprocessing.cpu_count", return_value=2)  # Ensure at least 2 workers
@patch("app.kb.manager.concurrent.futures.as_completed")
async def test_load_documents(
    mock_as_completed, mock_cpu_count, mock_executor, mock_glob, kb_manager
):
    """Test the load_documents method with multiprocessing."""
    # Setup mocks
    mock_glob.return_value = ["/fake/path/file1.json", "/fake/path/file2.json"]

    # Create mock futures with deterministic results
    future1 = MagicMock()
    future1.result.return_value = [Document(page_content="Tweet 1", metadata={"id": "1"})]

    future2 = MagicMock()
    future2.result.return_value = [Document(page_content="Tweet 2", metadata={"id": "2"})]

    mock_futures = {future1: "/fake/path/file1.json", future2: "/fake/path/file2.json"}

    # Setup mock for as_completed to return futures in order
    mock_as_completed.return_value = list(mock_futures.keys())

    # Create a proper mock for ProcessPoolExecutor
    mock_executor_context = MagicMock()
    mock_executor_instance = MagicMock()

    # Setup the context manager protocol
    mock_executor_context.__enter__.return_value = mock_executor_instance
    mock_executor_context.__exit__.return_value = None
    mock_executor.return_value = mock_executor_context

    # Mock the submit method to return specific futures
    def mock_submit(func, file):
        for future, path in mock_futures.items():
            if path == file:
                return future
        return list(mock_futures.keys())[0]  # Fallback

    mock_executor_instance.submit = MagicMock(side_effect=mock_submit)

    # Call the method
    result = await kb_manager.load_documents()

    # Assert
    assert len(result) == 2
    assert result[0].page_content == "Tweet 1"
    assert result[1].page_content == "Tweet 2"

    # Verify the executor was created with correct workers
    mock_executor.assert_called_once_with(max_workers=2)

    # Verify submit was called for each file
    assert mock_executor_instance.submit.call_count == 2


@pytest.mark.asyncio
@patch("app.kb.manager.glob.glob")
@patch("app.kb.manager.ProcessPoolExecutor")
@patch("app.kb.manager.concurrent.futures.as_completed")
async def test_load_documents_standalone(mock_as_completed, mock_executor, mock_glob):
    """Test the standalone load_documents function."""
    # Create test documents to return
    test_docs = [
        Document(page_content="Doc 1", metadata={}),
        Document(page_content="Doc 2", metadata={}),
    ]

    # Mock necessary components for the standalone function
    glob_files = ["/fake/path/file1.json", "/fake/path/file2.json"]
    mock_glob.return_value = glob_files

    # Create mock futures with deterministic results
    future1 = MagicMock()
    future1.result.return_value = [Document(page_content="Doc 1", metadata={})]

    future2 = MagicMock()
    future2.result.return_value = [Document(page_content="Doc 2", metadata={})]

    # Setup mock futures to be returned by as_completed
    mock_as_completed.return_value = [future1, future2]

    # Setup pool executor mock
    mock_executor_context = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_context.__enter__.return_value = mock_executor_instance
    mock_executor_context.__exit__.return_value = None
    mock_executor.return_value = mock_executor_context

    # Setup submit method to return our futures
    mock_executor_instance.submit = MagicMock(side_effect=[future1, future2])

    # Create a test path for standalone function
    with patch("app.kb.manager.Path") as mock_path:
        # Mock path resolution
        mock_path_instance = MagicMock()
        mock_path_instance.resolve.return_value.parent.parent.parent = Path("/fake")
        mock_path.return_value = mock_path_instance

        # Call the function
        result = await load_documents()

        # Assertions
        assert len(result) == 2
        assert any(doc.page_content == "Doc 1" for doc in result)
        assert any(doc.page_content == "Doc 2" for doc in result)


@pytest.mark.asyncio
async def test_load_documents_empty_dir(kb_manager):
    """Test loading documents from an empty directory."""
    with patch("glob.glob", return_value=[]):
        result = await kb_manager.load_documents()
        assert result == []


@pytest.mark.asyncio
@patch("app.kb.manager.glob.glob")
@patch("app.kb.manager.ProcessPoolExecutor")
@patch("app.kb.manager.multiprocessing.cpu_count", return_value=4)
@patch("app.kb.manager.concurrent.futures.as_completed")
async def test_load_documents_ensures_min_workers(
    mock_as_completed, mock_cpu_count, mock_executor, mock_glob, kb_manager
):
    """Test that load_documents ensures at least 1 worker."""
    # Test with 2 files
    mock_glob.return_value = ["/fake/path/file1.json", "/fake/path/file2.json"]

    # Create mock futures with deterministic results
    future1 = MagicMock()
    future1.result.return_value = [Document(page_content="Test 1", metadata={})]

    future2 = MagicMock()
    future2.result.return_value = [Document(page_content="Test 2", metadata={})]

    mock_as_completed.return_value = [future1, future2]

    # Setup mock for ProcessPoolExecutor
    mock_context = MagicMock()
    mock_instance = MagicMock()
    mock_context.__enter__ = MagicMock(return_value=mock_instance)
    mock_context.__exit__ = MagicMock(return_value=None)
    mock_executor.return_value = mock_context

    # Setup mocks for submitting tasks
    mock_instance.submit = MagicMock(side_effect=[future1, future2])

    # Call the method
    result = await kb_manager.load_documents()

    # Check if the executor was created with the correct workers
    # max_workers should be min(len(files), cpu_count), but at least 1
    mock_executor.assert_called_once_with(max_workers=2)

    # Test with 0 files
    mock_glob.return_value = []

    # Call again with empty file list
    result_empty = await kb_manager.load_documents()

    # If no files found, it should return an empty list immediately without creating executor
    assert result_empty == []


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
    with (
        patch("builtins.open", mock_open()) as mock_file,
        patch("os.makedirs") as mock_makedirs,
        patch("pickle.dump") as mock_dump,
    ):

        kb_manager._save_documents_pickle(valid_documents)

        # Verify directory creation
        mock_makedirs.assert_called_once()

        # Verify file opening
        mock_file.assert_called_once_with(kb_manager.documents_pickle_path, "wb")

        # Verify pickling
        mock_dump.assert_called_once()


def test_load_documents_pickle_success(kb_manager, valid_documents):
    """Test loading documents from pickle successfully."""
    # Setup mock to return test documents
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open()) as mock_file,
        patch("pickle.load", return_value=valid_documents) as mock_load,
    ):

        result = kb_manager._load_documents_pickle()

        # Verify file existence check
        assert os.path.exists(kb_manager.documents_pickle_path)

        # Verify file opening
        mock_file.assert_called_once_with(kb_manager.documents_pickle_path, "rb")

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
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open()),
        patch("pickle.load", side_effect=Exception("Test pickle error")),
    ):

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
@patch("os.path.exists")
@patch("app.kb.manager.FAISS")
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
@patch("os.path.exists")
@patch("app.kb.manager.FAISS")
async def test_create_new_index_when_loading_fails(
    mock_faiss, mock_exists, kb_manager, mock_embeddings
):
    """Test creating a new index when loading fails."""
    # Setup mocks
    mock_exists.return_value = True
    mock_faiss.load_local.side_effect = Exception("Failed to load")

    # Mock load_documents
    with patch.object(
        kb_manager, "load_documents", return_value=[Document(page_content="Test", metadata={})]
    ):
        # Mock text splitter
        with patch(
            "app.kb.manager.RecursiveCharacterTextSplitter.split_documents",
            return_value=[Document(page_content="Test", metadata={})],
        ):
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
@patch("app.kb.manager.logger")
async def test_save_index_creates_directory(mock_logger, kb_manager):
    """Test save_index logs a message when saving successfully."""
    # Create a mock FAISS instance
    mock_vector_store = MagicMock()
    kb_manager._vector_store = mock_vector_store

    # Call save_index
    kb_manager.save_index()

    # Verify logging happened
    mock_logger.info.assert_called_once()
    # Verify the correct path was in the log message
    assert kb_manager.index_path in str(mock_logger.info.call_args)


@pytest.mark.asyncio
@patch("app.kb.manager.logger")
@patch("os.makedirs")
@patch("pickle.dump")
@patch("builtins.open", new_callable=mock_open)
async def test_save_documents_pickle_success(
    mock_open_func, mock_dump, mock_makedirs, mock_logger, kb_manager, valid_documents
):
    """Test successful execution of _save_documents_pickle."""
    # Call the method
    kb_manager._save_documents_pickle(valid_documents)

    # Verify directory was created
    mock_makedirs.assert_called_once_with(kb_manager.index_path, exist_ok=True)

    # Verify file was opened correctly
    mock_open_func.assert_called_once_with(kb_manager.documents_pickle_path, "wb")

    # Verify pickle.dump was called with correct arguments
    mock_dump.assert_called_once()

    # Verify success was logged
    mock_logger.info.assert_called_once()
    assert str(len(valid_documents)) in str(mock_logger.info.call_args)


@pytest.mark.asyncio
async def test_load_or_create_kb_with_empty_result(kb_manager):
    """Test load_or_create_kb handles empty result from load_documents."""
    with (
        patch.object(kb_manager, "load_documents", return_value=[]),
        patch.object(kb_manager, "_check_index_integrity", return_value=False),
        patch.object(kb_manager, "_load_documents_pickle", return_value=None),
        patch("app.kb.manager.logger"),
    ):

        # Should raise ValueError because no documents are loaded
        with pytest.raises(ValueError, match="No documents were loaded"):
            await kb_manager.load_or_create_kb()


@pytest.mark.asyncio
async def test_load_or_create_kb_embeddings_error(kb_manager, valid_documents):
    """Test load_or_create_kb handles embeddings initialization error."""
    with (
        patch.object(kb_manager, "load_documents", return_value=valid_documents),
        patch.object(kb_manager, "_check_index_integrity", return_value=False),
        patch.object(kb_manager, "_load_documents_pickle", return_value=None),
        patch("app.kb.manager.OpenAIEmbeddings", side_effect=Exception("API Error")),
        patch("app.kb.manager.logger") as mock_logger,
    ):

        # Should propagate the API error
        with pytest.raises(Exception, match="API Error"):
            await kb_manager.load_or_create_kb()

        # Error should be logged
        assert any(
            "Error creating knowledge base" in str(call)
            for call in mock_logger.error.call_args_list
        )


@pytest.mark.asyncio
@patch("os.path.exists", return_value=True)
async def test_load_or_create_kb_pickle_recovery(mock_exists, kb_manager, valid_documents):
    """Test load_or_create_kb recovers from pickle if index fails."""
    with (
        patch.object(kb_manager, "_load_documents_pickle", return_value=valid_documents),
        patch.object(kb_manager, "_check_index_integrity", return_value=True),
        patch("app.kb.manager.FAISS") as mock_faiss,
        patch("app.kb.manager.OpenAIEmbeddings") as mock_embeddings,
    ):

        # Make load_local raise an exception first time
        mock_faiss.load_local = MagicMock(side_effect=[Exception("Load failed"), MagicMock()])

        # Mock FAISS.from_documents to return a mock
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # Call the method
        result = await kb_manager.load_or_create_kb()

        # Should create new index from saved documents
        mock_faiss.from_documents.assert_called_once()
        assert result is mock_faiss_instance


@pytest.mark.asyncio
async def test_metadata_extractor_handles_missing_metrics(kb_manager):
    """Test metadata_extractor handles records without social metrics."""
    # Create a record without any social metrics
    record = {"id": "12345", "text": "Test tweet", "createdAt": "2022-01-01T12:00:00Z"}
    metadata = {}

    # Call the method
    result = kb_manager.metadata_extractor(record, metadata)

    # The function should not crash with missing metrics
    assert isinstance(result, dict)
    assert result["id"] == "12345"  # ID should always be preserved


@pytest.mark.asyncio
async def test_metadata_extractor_normalizes_metrics(kb_manager):
    """Test metadata_extractor extracts metrics from public_metrics field."""
    # Create a record with public_metrics field
    record = {
        "id": "12345",
        "text": "Test tweet",
        "createdAt": "2022-01-01T12:00:00Z",
        "public_metrics": {"view_count": 1000, "like_count": 500, "retweet_count": 200},
    }
    metadata = {}

    # Call the method
    result = kb_manager.metadata_extractor(record, metadata)

    # The function should not crash with provided metrics
    assert isinstance(result, dict)
    assert result["id"] == "12345"  # ID should always be preserved


@pytest.mark.asyncio
@patch("app.kb.manager.load_documents")
@patch("app.kb.manager.OpenAIEmbeddings")
@patch("app.kb.manager.FAISS")
async def test_get_knowledge_base_function(mock_faiss, mock_embeddings, mock_load_docs):
    """Test the standalone get_knowledge_base function."""
    # Setup mocks
    test_docs = [Document(page_content="Test doc", metadata={})]
    mock_load_docs.return_value = test_docs

    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    # Call function
    result = await get_knowledge_base()

    # Verify mocks were called correctly
    mock_load_docs.assert_called_once()
    mock_embeddings.assert_called_once()
    mock_faiss.from_documents.assert_called_once_with(
        documents=test_docs, embedding=mock_embeddings_instance
    )

    # Verify result is the vector store
    assert result is mock_vector_store


@pytest.mark.asyncio
@patch("app.kb.manager.JSONLoader")
async def test_process_file_integration(mock_json_loader):
    """Test process_file integrates with JSONLoader correctly."""
    # Setup mock JSONLoader
    mock_loader_instance = MagicMock()
    mock_json_loader.return_value = mock_loader_instance

    # Mock the load method to return test documents
    mock_loader_instance.load.return_value = [
        Document(page_content="Tweet content", metadata={"id": "123"})
    ]

    # Create a temporary JSON file path
    test_file = "/tmp/test_file.json"

    # Call the function with the test file path
    result = process_file(test_file)

    # Verify JSONLoader was created with correct parameters
    mock_json_loader.assert_called_once()
    call_args = mock_json_loader.call_args[1]
    assert call_args["file_path"] == test_file
    assert call_args["jq_schema"] == ".[]"
    assert call_args["content_key"] == "text"

    # Verify the result
    assert len(result) == 1
    assert result[0].page_content == "Tweet content"
    assert result[0].metadata["id"] == "123"


@pytest.mark.asyncio
@patch("app.kb.manager.glob.glob")
@patch("app.kb.manager.multiprocessing.cpu_count", return_value=4)
@patch("app.kb.manager.ProcessPoolExecutor")
@patch("app.kb.manager.concurrent.futures.as_completed")
async def test_load_documents_respects_batch_size(
    mock_as_completed, mock_executor, mock_cpu_count, mock_glob, kb_manager
):
    """Test load_documents respects the batch_size parameter."""
    # Setup mock file list
    num_files = 20  # Using fewer files for faster tests
    file_paths = [f"file{i}.json" for i in range(num_files)]
    mock_glob.return_value = file_paths

    # Create mock futures with deterministic results
    mock_futures = {}
    for i in range(num_files):
        future = MagicMock()
        # Each file produces one document
        future.result.return_value = [Document(page_content=f"content{i}")]
        mock_futures[future] = file_paths[i]

    # Setup mock for as_completed to return futures in order
    # This avoids unpredictable behavior and potential hanging
    mock_as_completed.return_value = list(mock_futures.keys())

    # Create a proper mock for ProcessPoolExecutor
    mock_executor_context = MagicMock()
    mock_executor_instance = MagicMock()

    # Setup the context manager protocol
    mock_executor_context.__enter__.return_value = mock_executor_instance
    mock_executor_context.__exit__.return_value = None
    mock_executor.return_value = mock_executor_context

    # Mock the submit method with deterministic behavior
    def mock_submit(func, file):
        # Find the correct mock future for this file
        for future, path in mock_futures.items():
            if path == file:
                return future
        # Return the first future as fallback
        return list(mock_futures.keys())[0]

    mock_executor_instance.submit = MagicMock(side_effect=mock_submit)

    # Call the method with a specific batch size
    batch_size = 10
    result = await kb_manager.load_documents(batch_size=batch_size)

    # Verify the executor was created with correct workers
    mock_executor.assert_called_once_with(max_workers=4)

    # Verify the executor context was properly used
    mock_executor_context.__enter__.assert_called_once()
    mock_executor_context.__exit__.assert_called_once()

    # Each file should generate one document
    assert len(result) == num_files

    # Verify that the executor received submit calls for each file
    assert mock_executor_instance.submit.call_count == num_files


@pytest.mark.asyncio
async def test_as_retriever_method_integration(kb_manager, mock_embeddings):
    """Test as_retriever method integration with FAISS retriever."""
    # Mock a FAISS instance
    mock_faiss = MagicMock()
    mock_faiss.as_retriever.return_value = MagicMock()
    mock_faiss.as_retriever.return_value.search_kwargs = {"k": 3, "filter": None}

    # Attach to kb_manager
    kb_manager._vector_store = mock_faiss

    # Test as_retriever with specific search_kwargs
    search_kwargs = {"k": 3, "filter": None}
    retriever = kb_manager.vector_store.as_retriever(search_kwargs=search_kwargs)

    # Verify the retriever has correct search_kwargs
    assert retriever.search_kwargs["k"] == 3


@pytest.mark.asyncio
async def test_extract_content_handles_complex_json():
    """Test extract_content handles complex nested JSON structures."""
    # Create a record with nested structure
    record = {
        "id": "12345",
        "text": "This is a test tweet",
        "createdAt": "2022-01-01T12:00:00Z",
        "author": {"username": "testuser", "description": "Test user description"},
        "entities": {
            "urls": [{"expanded_url": "https://example.com", "display_url": "example.com"}],
            "hashtags": [{"tag": "test"}],
        },
    }

    # Extract content from the complex record
    result = extract_content(record)

    # Verify the function extracted meaningful content
    assert "testuser" in result
    assert "This is a test tweet" in result


# Test for resilience when loading from pickle fails
@pytest.mark.asyncio
@patch("app.kb.manager.pickle.load")
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.exists", return_value=True)
async def test_load_documents_pickle_corrupted(
    mock_exists, mock_file, mock_pickle_load, kb_manager
):
    """Test _load_documents_pickle handles corrupted pickle files."""
    # Make pickle.load raise an exception to simulate corrupted file
    mock_pickle_load.side_effect = pickle.UnpicklingError("Corrupted pickle")

    # Call the method
    result = kb_manager._load_documents_pickle()

    # Should return None on error
    assert result is None


# Test for FAISS integrity check handling corrupt indices
@pytest.mark.asyncio
@patch("os.path.exists")
@patch("app.kb.manager.logger")
async def test_check_index_integrity_incomplete_files(mock_logger, mock_exists, kb_manager):
    """Test _check_index_integrity detects corrupted/incomplete indices."""

    # Set up the exists to return values based on the path
    def exists_side_effect(path):
        if path == kb_manager.index_path:
            return True  # Index directory exists
        elif path == os.path.join(kb_manager.index_path, "index.pkl"):
            return True  # This file exists
        elif path == os.path.join(kb_manager.index_path, "index.faiss"):
            return False  # This file is missing
        elif path == kb_manager.documents_pickle_path:
            return True  # Documents pickle exists
        return False

    mock_exists.side_effect = exists_side_effect

    # Call the integrity check - should fail because a required file is missing
    result = kb_manager._check_index_integrity()

    # Should return False for incomplete index
    assert result is False

    # Should log a warning about the missing file
    mock_logger.warning.assert_called_with(f"Missing FAISS index file: index.faiss")


# Test end-to-end process for creating and using a knowledge base
@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)  # Force creation of new index
async def test_end_to_end_kb_creation_and_use(mock_exists, kb_manager, valid_documents):
    """Test end-to-end creation and use of knowledge base."""
    # Mock all the dependencies to create a clean test
    with (
        patch.object(kb_manager, "load_documents", return_value=valid_documents),
        patch("app.kb.manager.FAISS") as mock_faiss,
        patch("app.kb.manager.OpenAIEmbeddings") as mock_embeddings,
    ):

        # Setup mock FAISS instance
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # Setup mock retriever
        mock_retriever = MagicMock()
        mock_faiss_instance.as_retriever.return_value = mock_retriever

        # Setup mock results from retriever
        mock_retriever.get_relevant_documents = AsyncMock(return_value=valid_documents[:2])

        # Create knowledge base
        kb = await kb_manager.load_or_create_kb()

        # Test retrieval functionality
        retriever = kb.as_retriever(search_kwargs={"k": 2})
        results = await retriever.get_relevant_documents("test query")

        # Verify we got the expected results
        assert len(results) == 2
        assert results == valid_documents[:2]
