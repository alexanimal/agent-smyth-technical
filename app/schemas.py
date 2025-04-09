from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    """Types of queries supported by the system."""
    GENERAL = "general"
    INVESTMENT = "investment"
    TRADING = "trading"

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=3, # Validator handles stripping
        max_length=1000,
        description="The user's query text",
        json_schema_extra={'example': "What are investors saying about AAPL?"}
    )
    num_results: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of sources to retrieve"
    )
    query_type: Optional[QueryType] = Field(
        None,
        description="Override automatic query classification"
    )
    verbose: bool = Field(
        False,
        description="Whether to include detailed source information"
    )
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Optional context information for the query",
        json_schema_extra={'example': {"user_location": "US", "platform": "web"}}
    )
    model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="The model to use for the query",
        json_schema_extra={'example': "gpt-4o-mini"}
    )

    # Use model_config instead of inner Config class
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What are the recent trends for AAPL stock?",
                "num_results": 5,
                "context": {"session_id": "user123"},
                "model": "gpt-4o-mini"
            }
        }
    )

    @field_validator('message')
    @classmethod
    def message_must_be_meaningful(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError('Message must contain meaningful content (at least 3 chars)')
        return v

class ChatResponse(BaseModel):
    request_id: str
    response: str
    sources: List[str] = []
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )

    # Use model_config instead of inner Config class
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "Apple stock has shown positive momentum in recent weeks...",
                "sources": ["https://twitter.com/example/status/123456", "https://twitter.com/finance/status/789012"],
                "processing_time": 1.24,
                "timestamp": "2023-04-28T14:23:15.123456",
                "metadata": {"model_used": "gpt-4", "token_count": 423}
            }
        }
    )

class HealthStatus(BaseModel):
    status: str
    knowledge_base_loaded: bool
    is_loading: bool

class RootResponse(BaseModel):
    message: str
    status: str 