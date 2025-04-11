# RAG Module

The RAG (Retrieval Augmented Generation) module implements a sophisticated workflow for augmenting LLM responses with retrieved information. This module is the core of the knowledge retrieval and response generation system.

## Overview

The RAG module uses LangGraph to define a directed workflow graph for processing user queries through several stages:
1. Query classification
2. Document retrieval
3. Document ranking
4. Response generation
5. Alternative viewpoint generation (for certain query types)

## Components

### State Management (`state.py`)

Defines the `RAGState` TypedDict that flows through the workflow, containing:
- User query
- Classification information
- Retrieved documents
- Ranked documents
- Generated responses
- Source information
- Alternative viewpoints
- Processing metrics

### Workflow Graph (`graph.py`)

Configures the LangGraph workflow by:
- Creating a state graph with `RAGState`
- Adding processing nodes
- Defining edges and conditional routing
- Compiling the graph into a runnable workflow

### Processing Nodes (`nodes.py`)

Implements the core functionality of each step in the workflow:
- `classify_query_node`: Determines query type (technical, investment, trading, general)
- `retrieve_documents_node`: Retrieves relevant documents based on query type
- `rank_documents_node`: Ranks and diversifies retrieved documents
- `generate_response_node`: Produces primary response with source attribution
- `generate_alternative_node`: Creates alternative viewpoints for balance

### Document Scoring (`scoring.py`)

Provides utilities for ranking and diversifying documents:
- Social media engagement metrics
- Sentiment diversity
- Recency weighting
- Domain authority

### Utilities (`utils.py`)

Helper functions for the RAG components:
- LLM instance caching with `get_cached_llm`
- Fallback generation with `generate_with_fallback`
- Message content standardization

## Usage

The `app_workflow` compiled graph can be used to process queries:

```python
from app.rag.graph import app_workflow
from app.rag.state import RAGState

# Initialize state with user query
initial_state: RAGState = {
    "query": "What are investors saying about AAPL?",
    "num_results": 25,
    "model": "gpt-4-turbo",
    "ranking_config": {
        "recency_weight": 0.3,
        "view_weight": 0.2,
        "like_weight": 0.2,
        "retweet_weight": 0.3
    },
    "generate_alternative_viewpoint": True
}

# Run the workflow
result = await app_workflow.ainvoke(initial_state)
```

## Extending the Module

To add new functionality:
1. Define new node functions in `nodes.py`
2. Add them to the graph in `graph.py`
3. Update the `RAGState` in `state.py` if new data needs to flow through the graph

For custom document scoring, extend the functions in `scoring.py`.
