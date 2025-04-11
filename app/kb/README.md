# Knowledge Base Package

This package handles the creation, management, and retrieval of the vector knowledge base for the RAG Agent application.

## Contents

- `manager.py`: Implements the `KnowledgeBaseManager` class for loading, saving, and querying the knowledge base.
- `__init__.py`: Re-exports knowledge base components for backward compatibility.

## Usage

```python
from app.kb import KnowledgeBaseManager

# Initialize the knowledge base manager
kb_manager = KnowledgeBaseManager(mocks_dir="path/to/data")

# Load or create the knowledge base
kb = await kb_manager.load_or_create_kb()
```

## Key Features

- Document loading from JSON files with parallel processing
- FAISS vector store integration for efficient retrieval
- Metadata extraction from documents
- Caching of documents and embeddings for performance
- Support for incremental updates to the knowledge base
