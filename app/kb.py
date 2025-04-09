"""
Module for creating and managing the knowledge base from JSON files.
"""

import concurrent.futures
import glob
import logging
import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define chunking parameters as constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


class KnowledgeBaseManager:
    """Manages the creation, storage, and retrieval of the knowledge base."""

    mocks_dir: str
    index_path: str
    documents_pickle_path: str
    _vector_store: Optional[FAISS]

    def __init__(self, mocks_dir: Optional[str] = None):
        """
        Initialize the KnowledgeBaseManager with paths to data and indexes.

        Args:
            mocks_dir: Optional override for the mocks directory path (as string)
        """
        # Determine project root dynamically inside __init__
        project_root = Path(__file__).resolve().parent.parent

        if mocks_dir:
            self.mocks_dir = mocks_dir
        else:
            # Define mocks_dir relative to the dynamically found project root
            self.mocks_dir = str(project_root / "data")

        # Log the resolved path to help with debugging
        print(f"Using mocks directory: {self.mocks_dir}")

        # Define paths relative to the dynamically found project root
        self.index_path = str(project_root / "faiss_index")
        self.documents_pickle_path = str(Path(self.index_path) / "documents.pkl")
        self._vector_store = None

    @staticmethod
    def metadata_extractor(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a tweet record."""
        return {
            "id": record.get("id", ""),
            "created_at": record.get("createdAt", ""),
            "url": record.get("url", ""),
            "profile": record.get("profile", ""),
            "tweet_time": record.get("tweet_time", ""),
        }

    @staticmethod
    def process_file(json_file_path: str) -> List[Document]:
        """Process a single JSON file and return documents."""
        try:
            loader = JSONLoader(
                file_path=json_file_path,
                jq_schema=".[]",
                content_key="text",
                metadata_func=KnowledgeBaseManager.metadata_extractor,
            )
            return loader.load()
        except Exception as e:
            print(f"Error processing {json_file_path}: {str(e)}")
            return []

    async def load_documents(self, batch_size: int = 1000) -> List[Document]:
        """Load documents using parallel processing."""
        json_files = glob.glob(os.path.join(self.mocks_dir, "*.json"))

        print(f"Found {len(json_files)} JSON files in {self.mocks_dir}")

        # Use multiprocessing.Pool for better handling of large files
        cpu_count = multiprocessing.cpu_count()
        # Ensure at least 1 process is used, even if no files are found
        num_processes = max(1, min(cpu_count, len(json_files)))
        print(f"Using {num_processes} processes to load documents")

        # If no files found, return empty list early
        if not json_files:
            print(f"Warning: No JSON files found in {self.mocks_dir}")
            return []

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(self.process_file, file): file for file in json_files}
            all_documents = []

            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    results = future.result()
                    all_documents.extend(results)
                    print(f"Successfully processed {len(results)} documents from {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {str(e)}")

        return all_documents

    @property
    def vector_store(self) -> Optional[FAISS]:
        """Get the vector store, loading it if necessary."""
        return self._vector_store

    def save_index(self) -> None:
        """Save the index to disk."""
        if self._vector_store:
            # This saves the entire vector store including embeddings
            self._vector_store.save_local(self.index_path)
            logger.info(f"Index saved to {self.index_path}")

    def _save_documents_pickle(self, documents: List[Document]) -> None:
        """
        Save documents to a pickle file to avoid re-embedding.

        Args:
            documents: List of processed documents
        """
        # Ensure directory exists
        os.makedirs(self.index_path, exist_ok=True)

        # Save the documents to pickle
        with open(self.documents_pickle_path, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Saved {len(documents)} documents to {self.documents_pickle_path}")

    def _load_documents_pickle(self) -> Optional[List[Document]]:
        """
        Load documents from pickle file if available.

        Returns:
            List of documents or None if pickle doesn't exist
        """
        if not os.path.exists(self.documents_pickle_path):
            return None

        try:
            with open(self.documents_pickle_path, "rb") as f:
                documents = pickle.load(f)
            logger.info(f"Loaded {len(documents)} documents from {self.documents_pickle_path}")

            # Calculate total text size
            total_text_size = sum(len(doc.page_content) for doc in documents)
            logger.info(f"Total document content size: {total_text_size} characters")

            # Count documents by source
            sources: Dict[str, int] = {}
            for doc in documents:
                source = doc.metadata.get("url", "unknown")
                sources[source] = sources.get(source, 0) + 1
            logger.info(f"Documents by source: {len(sources)} unique sources")

            return documents
        except Exception as e:
            logger.error(f"Error loading documents pickle: {str(e)}")
            return None

    def _check_index_integrity(self) -> bool:
        """
        Check if the FAISS index and documents pickle are valid.

        Returns:
            True if both components exist and appear valid
        """
        # Check if index directory exists
        if not os.path.exists(self.index_path):
            return False

        # Check for required FAISS index files
        index_files = ["index.faiss", "index.pkl"]
        for file in index_files:
            if not os.path.exists(os.path.join(self.index_path, file)):
                logger.warning(f"Missing FAISS index file: {file}")
                return False

        # Check if documents pickle exists
        if not os.path.exists(self.documents_pickle_path):
            logger.warning("Missing documents pickle file")
            return False

        return True

    async def load_or_create_kb(self) -> FAISS:
        """Load an existing knowledge base or create a new one."""
        # Try to load the existing index first
        if self._check_index_integrity():
            try:
                logger.info(f"Loading existing index from {self.index_path}...")
                # This loads the complete vector store with embeddings
                self._vector_store = FAISS.load_local(
                    self.index_path,
                    embeddings=OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True,
                )
                logger.info("Successfully loaded existing index")
                return self._vector_store
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

        # Try loading just the documents if available to avoid reprocessing
        cached_documents = self._load_documents_pickle()

        if cached_documents:
            try:
                logger.info("Creating new index from cached documents...")
                # Use the defined chunking parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                )

                split_documents = text_splitter.split_documents(cached_documents)
                logger.info(f"Split cached documents into {len(split_documents)} chunks")

                # Create new vector store
                embeddings = OpenAIEmbeddings()
                self._vector_store = FAISS.from_documents(
                    documents=split_documents, embedding=embeddings
                )

                # Save the new index
                self.save_index()

                # Save the chunked documents instead
                self._save_documents_pickle(split_documents)

                return self._vector_store
            except Exception as e:
                logger.error(f"Error creating index from cached documents: {str(e)}")
                # Continue to full processing if this fails

        # Create new index from scratch
        try:
            logger.info("Creating new knowledge base from source files...")
            documents = await self.load_documents()

            if not documents:
                raise ValueError("No documents were loaded")

            # Save documents pickle for future use
            self._save_documents_pickle(documents)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )

            split_documents = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_documents)} chunks")

            embeddings = OpenAIEmbeddings()
            self._vector_store = FAISS.from_documents(
                documents=split_documents, embedding=embeddings
            )

            # Save the new index
            self.save_index()

            # Save the chunked documents instead
            self._save_documents_pickle(split_documents)

            return self._vector_store
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            raise

    def validate_documents(self, documents: List[Document]) -> bool:
        """Validate that loaded documents meet expected criteria."""
        if not documents:
            print("No documents loaded!")
            return False

        # Check minimum document count (adjust as needed)
        if len(documents) < 1000:  # Adjust expected minimum
            print(f"Warning: Only {len(documents)} documents loaded, expected at least 1000")
            return False

        # Check for unique sources
        sources = set(doc.metadata.get("url", "") for doc in documents if "url" in doc.metadata)
        if len(sources) < 10:  # Adjust expected minimum sources
            print(f"Warning: Only {len(sources)} unique sources found, expected at least 10")
            return False

        print(f"Document validation passed: {len(documents)} documents with {len(sources)} sources")
        return True


# Function to extract relevant content from the JSON documents
def extract_content(record: Dict[str, Any]) -> str:
    """
    Extract the relevant content from a tweet JSON record.

    Args:
        record: A dictionary containing tweet data

    Returns:
        A string containing the extracted content
    """
    # Extract data based on the actual structure found in the JSON files
    tweet_id = record.get("id", "")
    tweet_text = record.get("text", "")
    created_at = record.get("createdAt", "")

    # Try to get username if available
    username = "Unknown"
    if "author" in record and isinstance(record["author"], dict):
        username = record["author"].get("username", "Unknown")
    elif "profile" in record:
        username = str(record.get("profile", "Unknown"))

    # Additional context if available
    retweet_count = record.get("retweetCount", 0)
    like_count = record.get("likeCount", 0)
    reply_count = record.get("replyCount", 0)

    return (
        f"Tweet ID: {tweet_id} | Posted by: {username} on {created_at} | Tweet: {tweet_text} | "
        f"Stats: {retweet_count} retweets, {like_count} likes, {reply_count} replies"
    )


def metadata_extractor(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a tweet record.

    Args:
        record: The JSON record
        metadata: Any existing metadata

    Returns:
        A dictionary of metadata
    """
    return {
        "id": record.get("id", ""),
        "created_at": record.get("createdAt", ""),
        "url": record.get("url", ""),
        "profile": record.get("profile", ""),
        "tweet_time": record.get("tweet_time", ""),
    }


# Function to process a single file
def process_file(json_file_path):
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema=".[]",
        content_key="text",
        metadata_func=metadata_extractor,
    )
    return loader.load()


async def load_documents(batch_size: int = 1000) -> List[Document]:
    """Load all JSON documents using multiprocessing."""
    # Determine mocks directory path independently for this standalone function
    project_root = Path(__file__).resolve().parent.parent
    mocks_dir_path = project_root / "data"

    print(f"Using mocks directory (standalone): {mocks_dir_path}")
    # Explicitly convert Path to str for os.path.join
    json_files = glob.glob(os.path.join(str(mocks_dir_path), "*.json"))
    all_documents = []

    print(f"Found {len(json_files)} JSON files in {mocks_dir_path}")

    # Determine optimal number of processes
    cpu_count = multiprocessing.cpu_count()
    # Ensure at least 1 process is used, even if no files are found
    num_processes = max(1, min(cpu_count, len(json_files)))
    print(f"Using {num_processes} processes to load documents")

    # If no files found, return empty list early
    if not json_files:
        print(f"Warning: No JSON files found in {mocks_dir_path}")
        return []

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_file, file): file for file in json_files}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                print(f"Successfully processed {len(documents)} documents from {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    return all_documents


async def get_knowledge_base():
    """
    Create or load the knowledge base in memory.

    Returns:
        A FAISS vector store containing the embedded documents
    """
    # Load documents
    documents = await load_documents()

    if not documents:
        raise ValueError("No documents were loaded!")

    print(f"Loaded {len(documents)} documents in total.")

    # Use the defined chunking parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    split_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(split_documents)} chunks.")

    # Create the in-memory vector store
    vector_store = FAISS.from_documents(documents=split_documents, embedding=OpenAIEmbeddings())

    return vector_store
