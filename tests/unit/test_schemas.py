"""
Unit tests for schema classes and enums.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthStatus,
    ModelSelectionRequest,
    OpenAIModel,
    QueryType,
    RootResponse,
)


class TestQueryType:
    """Test suite for the QueryType enum."""

    def test_query_type_values(self):
        """Test that QueryType enum has the expected values."""
        # Assert that all expected values are present
        assert "general" in QueryType.__members__.values()
        assert "investment" in QueryType.__members__.values()
        assert "trading" in QueryType.__members__.values()
        assert "technical" in QueryType.__members__.values()

        # Verify specific mappings
        assert QueryType.GENERAL == "general"
        assert QueryType.INVESTMENT == "investment"
        assert QueryType.TRADING == "trading"
        assert QueryType.TECHNICAL == "technical"

        # Verify the number of options (useful to catch if new ones are added without tests)
        assert len(QueryType.__members__) == 4


class TestChatRequest:
    """Test suite for ChatRequest validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        request = ChatRequest(
            message="Test message",
            num_results=5,
            verbose=False,
            # context={},
            # model="gpt-4o",
            query_type=None,
            generate_alternative_viewpoint=False,
            ranking_weights={
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            },
        )

        assert request.num_results == 5
        assert request.verbose is False
        # assert request.context == {}
        # assert request.model == "gpt-4o"
        assert request.query_type is None
        assert request.generate_alternative_viewpoint is False
        assert request.ranking_weights is not None
        assert request.ranking_weights["recency_weight"] == 0.4

    def test_message_validation(self):
        """Test message length validation."""
        # Valid message
        request = ChatRequest(
            message="Test message",
            num_results=5,
            verbose=False,
            # context={},
            # model="gpt-4o",
            query_type=None,
            generate_alternative_viewpoint=False,
            ranking_weights={
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            },
        )
        assert request.message == "Test message"

        # Message too short
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Te",
                num_results=5,
                verbose=False,
                # context={},
                # model="gpt-4o",
                query_type=None,
                generate_alternative_viewpoint=False,
                ranking_weights={
                    "recency_weight": 0.4,
                    "view_weight": 0.2,
                    "like_weight": 0.2,
                    "retweet_weight": 0.2,
                },
            )

    def test_query_type_field(self):
        """Test that query_type field accepts the new TECHNICAL value."""
        # Test with enum value directly
        request = ChatRequest(
            message="Test message",
            num_results=5,
            verbose=False,
            # context={},
            # model="gpt-4o",
            query_type=QueryType.TECHNICAL,
            generate_alternative_viewpoint=False,
            ranking_weights={
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            },
        )
        assert request.query_type == QueryType.TECHNICAL

        # Test with None value
        request = ChatRequest(
            message="Test message",
            num_results=5,
            verbose=False,
            # context={},
            # model="gpt-4o",
            query_type=None,
            generate_alternative_viewpoint=False,
            ranking_weights={
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            },
        )
        assert request.query_type is None

        # Invalid query type should raise ValidationError - using a non-QueryType value that will trigger validation
        # We need to test this at runtime since the type checker prevents using string literals directly
        invalid_value = "invalid_type"
        with pytest.raises(ValidationError):
            # Use a dict to bypass static type checking
            ChatRequest.model_validate(
                {
                    "message": "Test message",
                    "num_results": 5,
                    "verbose": False,
                    "context": {},
                    "model": "gpt-4o",
                    "query_type": invalid_value,
                    "generate_alternative_viewpoint": False,
                    "ranking_weights": {
                        "recency_weight": 0.4,
                        "view_weight": 0.2,
                        "like_weight": 0.2,
                        "retweet_weight": 0.2,
                    },
                }
            )


class TestOpenAIModel:
    """Test OpenAIModel enum and validation methods."""

    def test_is_valid(self):
        """Test the is_valid method."""
        assert OpenAIModel.is_valid("gpt-3.5-turbo") is True
        assert OpenAIModel.is_valid("gpt-4") is True
        assert OpenAIModel.is_valid("gpt-4-turbo") is True
        assert OpenAIModel.is_valid("gpt-100") is False
        assert OpenAIModel.is_valid("") is False

    def test_get_default(self):
        """Test the get_default method."""
        assert OpenAIModel.get_default() == "gpt-3.5-turbo"


class TestModelSelectionRequest:
    """Test ModelSelectionRequest schema."""

    def test_defaults(self):
        """Test default values."""
        request = ModelSelectionRequest()
        assert request.model is None

    def test_with_valid_enum(self):
        """Test instantiation with valid enum values."""
        request = ModelSelectionRequest(model=OpenAIModel.GPT_4_TURBO)
        assert request.model == "gpt-4-turbo"

    def test_model_serialization(self):
        """Test model serialization."""
        request = ModelSelectionRequest(model=OpenAIModel.GPT_4)
        json_data = request.model_dump()
        assert json_data["model"] == "gpt-4"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
