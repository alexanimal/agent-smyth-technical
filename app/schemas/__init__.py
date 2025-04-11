"""
Schemas package for the RAG Agent application.

This package re-exports Pydantic models from the api module
to maintain backward compatibility.
"""

from app.schemas.api import ChatRequest, ChatResponse, HealthStatus, QueryType, RootResponse
from app.schemas.models import ModelSelectionRequest, OpenAIModel

# For backward compatibility
__all__ = [
    "QueryType",
    "ChatRequest",
    "ChatResponse",
    "HealthStatus",
    "RootResponse",
    "ModelSelectionRequest",
    "OpenAIModel",
]
