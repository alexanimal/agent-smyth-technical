# Core Package

This package contains the core functionality of the RAG Agent application, including the primary handler classes, routers, and models that power the RAG pipeline.

## Contents

- `handler.py`: Contains the `ChatHandler` class which manages the RAG pipeline, including document retrieval, source extraction, and response generation.
- `router.py`: Implements the query classification router that determines the type of query and routes it to the appropriate processing pipeline.
- `models.py`: Defines data models and schemas used throughout the application.
- `knowledge_base.py`: Contains the knowledge base implementation for document retrieval and vector store interactions.
- `__init__.py`: Re-exports core components for convenient imports.

## Usage

```python
from app.core import ChatHandler, Router, KnowledgeBase
from app.core.models import QueryType, ChatMessage

# Initialize core components
knowledge_base = KnowledgeBase("path_to_embeddings")
router = Router()
handler = ChatHandler(knowledge_base=knowledge_base)

# Process a user query
query = "What is the investment thesis for Google?"
query_type = router.classify_query(query)
response = handler.process_query(query, query_type=query_type)
```

## Key Features

- Query classification for routing to specialized processing pipelines
- Document retrieval from vector stores
- Source extraction and attribution
- Response generation with context
- Caching mechanisms for improved performance
- Error handling and retry mechanisms
