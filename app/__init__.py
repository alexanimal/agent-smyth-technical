"""
RAG Agent using FastAPI, LangChain, and OpenAI for processing tweet data.

This package implements a Retrieval-Augmented Generation (RAG) agent API
that allows querying knowledge bases built from tweet data. The agent
uses OpenAI models and LangChain components to understand queries
and retrieve relevant information.

Modules:
    chat: Contains the chat routing and handling components
    config: Configuration management and environment variables
    kb: Knowledge base implementation and indexing logic
    main: FastAPI application and endpoint definitions
    middleware: Request/response middleware components
    prompts: System prompts and templates
    schemas: Pydantic models and schema definitions
    services: Utility services and helper functions
"""
