"""
RAG Agent using FastAPI, LangChain, and OpenAI for processing tweet data.

This package implements a Retrieval-Augmented Generation (RAG) agent API
that allows querying knowledge bases built from tweet data. The agent
uses OpenAI models and LangChain components to understand queries
and retrieve relevant information.

Packages:
    config: Configuration management and environment variables
    core: Core components for RAG functionality
    kb: Knowledge base implementation and indexing logic
    middleware: Request/response middleware components
    prompts: System prompts and templates
    rag: RAG workflow and processing nodes
    routers: API endpoints and routing logic
    schemas: Pydantic models and schema definitions
    services: Utility services and helper functions
    utils: Utility functions for document processing and more
"""

# Re-export for backward compatibility
from app.config import settings, setup_logging_and_sentry
from app.kb import KnowledgeBaseManager, get_knowledge_base

__version__ = "1.0.0"
