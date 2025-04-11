# Utils Package

This package contains utility functions and helper modules that are used throughout the RAG Agent application. It provides reusable functionality for document processing, technical analysis, and other common operations.

## Contents

### Document Processing (`document.py`)

Provides utilities for working with documents in the RAG system:

- `extract_sources`: Extracts and deduplicates source URLs from document metadata
- `analyze_sentiment`: Performs sentiment analysis on documents for diversity ranking
- `diversify_documents`: Re-ranks documents to balance recency and viewpoint diversity

Usage example:

```python
from app.utils.document import extract_sources, diversify_documents
from langchain_core.documents import Document

# Extract sources from documents
docs = [Document(page_content="...", metadata={"url": "https://example.com"})]
sources = extract_sources(docs)

# Diversify documents for a balanced perspective
docs_with_timestamps = [(doc, timestamp) for doc, timestamp in zip(docs, [1234567890])]
diverse_docs = await diversify_documents(docs_with_timestamps, k=5)
```

### Technical Analysis (`technical.py`)

Utilities for enhancing responses to financial and technical analysis queries:

- `get_technical_indicators`: Extracts technical indicators and chart patterns from documents

Usage example:

```python
from app.utils.technical import get_technical_indicators
from langchain_core.documents import Document

# Get technical indicators from a query and documents
query = "What's the current RSI for $AAPL?"
docs = [Document(page_content="AAPL's RSI-14 is showing oversold conditions")]
technical_data = await get_technical_indicators(query, docs)
```

## Key Features

- **Document Processing**: Tools for handling, ranking, and diversifying retrieved documents
- **Source Extraction**: Utilities for extracting and formatting source information
- **Sentiment Analysis**: Basic sentiment analysis for document diversity
- **Technical Analysis**: Extraction of financial indicators and patterns from text
- **Error Handling**: Consistent patterns for error handling

## Usage in RAG Workflow

The utilities are primarily used within the RAG workflow to enhance retrieval and generation:

```python
from app.utils.document import extract_sources
from app.utils.technical import get_technical_indicators

async def process_documents(query, documents):
    # Extract technical information if relevant
    technical_data = await get_technical_indicators(query, documents)

    # Extract sources for attribution
    sources = extract_sources(documents)

    return {
        "sources": sources,
        "technical_data": technical_data
    }
```

## Extending Utilities

To add new utility functions:

1. Add the function to the appropriate module based on its purpose
2. Ensure it follows the same error handling patterns
3. Add appropriate logging for debugging
4. Update this README with documentation for the new functionality

For more complex additions, consider creating a new module within the utils package.
