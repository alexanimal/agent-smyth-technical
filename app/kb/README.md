# Knowledge Base Module

This module manages the storage, retrieval, and maintenance of the document knowledge base that powers the RAG system.

## Overview

The Knowledge Base (KB) module is responsible for:
- Loading and processing tweet data
- Creating and managing document embeddings
- Providing efficient vector-based retrieval
- Persisting the knowledge base to disk
- Background loading with progress tracking
- Verifying data integrity

## Components

### KnowledgeBaseManager

The core class that manages all KB operations:

```python
from app.kb import KnowledgeBaseManager

# Create a KB manager
kb_manager = KnowledgeBaseManager(mocks_dir="data")

# Load or create the knowledge base
kb = await kb_manager.load_or_create_kb()

# Retrieve documents by semantic search
results = kb.search("What are investors saying about AAPL?", k=10)
```

### Document Processing Pipeline

The module implements a processing pipeline for tweet data:

1. **Loading**: Reads JSON files from the data directory
2. **Chunking**: Splits documents into appropriate-sized chunks
3. **Embedding**: Generates vector embeddings for each chunk
4. **Indexing**: Stores embeddings in a FAISS vector store
5. **Persistence**: Saves the processed data to disk

## Performance Optimizations

Several optimizations improve performance and efficiency:

- **Parallel Processing**: Uses multiprocessing to speed up document loading
- **Batched Embedding**: Groups documents for efficient API usage
- **Persistent Cache**: Saves embeddings to avoid regeneration
- **Lazy Loading**: Loads the KB in the background during application startup
- **Vector Quantization**: Optimizes FAISS index for memory usage and performance

## File Structure

The KB data is stored in the following structure:

```
faiss_index/
  ├── index.faiss       # FAISS vector index file
  ├── documents.pkl     # Serialized document metadata
  ├── config.json       # Configuration and version info
  └── stats.json        # Statistics about the knowledge base
```

## Usage Examples

### Initialization

```python
from app.kb import KnowledgeBaseManager
from app.config import settings

# Initialize the KB manager
kb_manager = KnowledgeBaseManager(
    mocks_dir=settings.data_dir,
    index_dir=settings.index_dir
)

# Load or create the KB (runs in background if is_async=True)
kb = await kb_manager.load_or_create_kb(is_async=True)
```

### Document Retrieval

```python
# Search for relevant documents
results = kb.search(
    query="What are investors saying about AAPL?",
    k=25,  # Number of documents to retrieve
    filter={"date_range": ["2023-01-01", "2023-12-31"]}  # Optional filters
)

# Access document content and metadata
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Score: {doc.metadata['score']}")
    print(f"Source: {doc.metadata['url']}")
```

### Extending the Knowledge Base

```python
# Create new documents
from langchain_core.documents import Document

new_documents = [
    Document(
        page_content="Tesla stock has been rising after positive earnings.",
        metadata={"url": "https://example.com/news1", "timestamp": 1673452800}
    ),
    Document(
        page_content="AAPL announced a new product line today.",
        metadata={"url": "https://example.com/news2", "timestamp": 1673539200}
    )
]

# Add documents to the KB
await kb_manager.add_documents(new_documents)

# Save the updated KB
await kb_manager.save_kb()
```

### Health Check and Statistics

```python
# Verify KB integrity
is_healthy = await kb_manager.health_check()

# Get KB statistics
stats = kb_manager.get_stats()
print(f"Document count: {stats['document_count']}")
print(f"Embedding dimensions: {stats['embedding_dimensions']}")
print(f"Last updated: {stats['last_updated']}")
```

## Configuration Options

The KB manager can be configured with several options:

- **Embedding Model**: Select the embedding model to use (default: OpenAI)
- **Chunk Size**: Control the size of document chunks (default: 512 tokens)
- **Chunk Overlap**: Set the overlap between chunks (default: 50 tokens)
- **Vector Store Type**: Choose the vector store implementation (default: FAISS)
- **Similarity Metric**: Select the similarity function (default: cosine)

Example:

```python
from app.kb import KnowledgeBaseManager
from langchain_openai import OpenAIEmbeddings

kb_manager = KnowledgeBaseManager(
    mocks_dir="data",
    index_dir="faiss_index",
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-large"),
    chunk_size=1024,
    chunk_overlap=100,
    similarity_metric="l2"
)
```

## Error Handling and Recovery

The KB module includes robust error handling:

- **Integrity Verification**: Checks data integrity before loading
- **Automatic Recovery**: Rebuilds the KB if corruption is detected
- **Versioning**: Tracks KB version for compatibility
- **Graceful Degradation**: Falls back to simpler retrieval if advanced features fail

## Performance Considerations

For optimal performance:

- Keep document chunks between 512-1024 tokens
- Use batched addition for multiple documents
- Consider index quantization for large knowledge bases
- Monitor embedding API usage costs
- Use filters to narrow search scope when possible
