"""
Pydantic schema models for the RAG Agent application.

This module defines the data models used throughout the application for validating,
serializing, and deserializing data between the API and client applications. It includes
models for chat requests and responses, system status, and query type enumerations.

The models leverage Pydantic for automatic validation, documentation generation,
and JSON Schema compatibility.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.schemas.models import OpenAIModel


class QueryType(str, Enum):
    """
    Types of queries supported by the system.

    Enumeration of query types that determine how the system processes
    and responds to user queries.

    Attributes:
        GENERAL: General knowledge query type for standard RAG responses
        INVESTMENT: Investment-specific query type for financial analysis
        TRADING: Trading-related query type for market strategy queries
        TECHNICAL: Technical analysis query type for chart patterns and indicators
    """

    GENERAL = "general"
    INVESTMENT = "investment"
    TRADING = "trading"
    TECHNICAL = "technical"


class ChatRequest(BaseModel):
    """
    Model representing a chat request from a client.

    This model validates incoming chat requests, ensuring they meet
    the required format and constraints before processing. It includes
    the user's message, configuration options, and contextual information.

    Attributes:
        message: The user's query text to be processed
        num_results: Number of sources to retrieve and include
        query_type: Optional override for automatic query classification
        verbose: Whether to include detailed source information
        generate_alternative_viewpoint: Whether to generate an alternative perspective
        ranking_weights: Optional configuration for ranking document sources
        context: Optional context information for the query
        model: The LLM model to use for processing the query
    """

    message: str = Field(
        ...,
        min_length=3,  # Validator handles stripping
        max_length=50000,  # Increased limit to support large text inputs
        description="The user's query text",
        json_schema_extra={"example": "What are investors saying about AAPL?"},
    )
    num_results: int = Field(
        25,  # Default of 25 documents
        ge=3,  # At least 3 documents for meaningful analysis
        le=250,  # Upper limit of 100 for performance reasons
        description="Number of sources to retrieve and consider in analysis. Higher values provide more comprehensive analysis but may increase processing time. Recommended ranges: 3-10 for quick queries, 10-25 for detailed analysis, 25-100 for comprehensive market research.",
    )
    query_type: Optional[QueryType] = Field(
        None, description="Override automatic query classification"
    )
    verbose: bool = Field(False, description="Whether to include detailed source information")
    generate_alternative_viewpoint: bool = Field(
        False, description="Whether to generate an alternative perspective or counter-argument"
    )
    ranking_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional weights for document ranking signals. Keys: recency_weight, view_weight, like_weight, retweet_weight. Values should be between 0 and 1, and ideally sum to 1.",
        json_schema_extra={
            "example": {
                "recency_weight": 0.3,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.3,
            }
        },
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context information for the query",
        json_schema_extra={"example": {"user_location": "US", "platform": "web"}},
    )
    model: Optional[OpenAIModel] = Field(
        default=None,
        description="The OpenAI model to use for generating the response. If not specified, the default model will be used.",
        json_schema_extra={"example": "gpt-3.5-turbo"},
    )

    # Use model_config instead of inner Config class
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What are the recent trends for AAPL stock?",
                "num_results": 25,
                "ranking_weights": {
                    "recency_weight": 0.3,
                    "view_weight": 0.2,
                    "like_weight": 0.2,
                    "retweet_weight": 0.3,
                },
                "context": {"session_id": "user123"},
                "model": "gpt-3.5-turbo",
            }
        }
    )

    @field_validator("message")
    @classmethod
    def message_must_be_meaningful(cls, v: str) -> str:
        """
        Validates that the message contains meaningful content.

        Args:
            v: The message string to validate

        Returns:
            The validated message string

        Raises:
            ValueError: If the stripped message is less than 3 characters
        """
        if len(v.strip()) < 3:
            raise ValueError("Message must contain meaningful content (at least 3 chars)")
        return v

    @field_validator("ranking_weights")
    @classmethod
    def validate_ranking_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Validates the ranking weights.

        Args:
            v: Dictionary of ranking weights

        Returns:
            Validated ranking weights

        Raises:
            ValueError: If weights are invalid
        """
        if v is None:
            return v

        valid_keys = {"recency_weight", "view_weight", "like_weight", "retweet_weight"}

        # Validate keys
        for key in v:
            if key not in valid_keys:
                raise ValueError(f"Invalid ranking weight key: {key}. Valid keys are: {valid_keys}")

        # Validate values
        for key, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Weight for {key} must be between 0 and 1")

        # Check if weights sum to approximately 1
        weight_sum = sum(v.values())
        if not 0.98 <= weight_sum <= 1.02:  # Allow small rounding errors
            raise ValueError(f"Weights should sum to approximately 1, got {weight_sum}")

        return v


class ChatResponse(BaseModel):
    """
    Model representing a response to a chat request.

    This model structures the data returned to clients after processing
    a chat request, including the generated response, source attribution,
    and metadata about the processing.

    Attributes:
        request_id: Unique identifier for the request
        response: The generated text response to the user's query
        sources: List of source URLs that contributed to the response
        processing_time: Time taken to process the request in seconds
        timestamp: When the response was generated
        alternative_viewpoints: Optional counter-narrative to provide balanced perspective
        metadata: Additional metadata about the response and processing
    """

    request_id: str
    response: str
    sources: List[str] = []
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    alternative_viewpoints: Optional[str] = Field(
        None, description="Alternative perspective or counter-argument for balanced view"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )

    # Use model_config instead of inner Config class
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "Apple stock has shown positive momentum in recent weeks...",
                "sources": [
                    "https://twitter.com/example/status/123456",
                    "https://twitter.com/finance/status/789012",
                ],
                "processing_time": 1.24,
                "timestamp": "2023-04-28T14:23:15.123456",
                "alternative_viewpoints": "While Apple shows positive momentum, there are risks of market saturation...",
                "metadata": {
                    "model_used": "gpt-4",
                    "token_count": 423,
                    "query_type": "investment",
                    "confidence_scores": {
                        "investment": 75,
                        "technical": 20,
                        "trading_thesis": 5,
                        "general": 0,
                    },
                },
            }
        }
    )


class HealthStatus(BaseModel):
    """
    Model representing the system's health status.

    This model provides information about the current state of the system,
    including whether core components like the knowledge base are ready.

    Attributes:
        status: Current status description (e.g., "healthy", "initializing")
        knowledge_base_loaded: Whether the knowledge base is fully loaded
        is_loading: Whether the system is currently initializing components
    """

    status: str
    knowledge_base_loaded: bool
    is_loading: bool


class RootResponse(BaseModel):
    """
    Model representing the response from the root endpoint.

    This simple model provides basic information when accessing the API root.

    Attributes:
        message: A welcome or informational message
        status: The overall system status
    """

    message: str
    status: str
