"""
Unit tests for schema classes and enums.
"""

from datetime import datetime
from typing import Optional, cast  # Add typing imports for type checking fixes

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
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )

        assert request.num_results == 25  # Default is 25
        assert request.verbose is False
        assert request.query_type is None
        assert request.generate_alternative_viewpoint is False
        assert request.ranking_weights is None
        assert request.context is None
        assert request.model is None

    def test_message_validation(self):
        """Test message length validation."""
        # Valid message
        request = ChatRequest(
            message="Test message",
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.message == "Test message"

        # Message too short
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Te",
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
                ranking_weights=None,
            )

        # Message with whitespace that becomes too short
        with pytest.raises(ValidationError):
            ChatRequest(
                message="   A   ",
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
                ranking_weights=None,
            )

    def test_query_type_field(self):
        """Test that query_type field accepts the new TECHNICAL value."""
        # Test with enum value directly
        request = ChatRequest(
            message="Test message",
            query_type=QueryType.TECHNICAL,
            num_results=25,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.query_type == QueryType.TECHNICAL

        # Test with None value
        request = ChatRequest(
            message="Test message",
            query_type=None,
            num_results=25,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
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
                    "query_type": invalid_value,
                    "num_results": 25,
                    "verbose": False,
                    "generate_alternative_viewpoint": False,
                    "ranking_weights": None,
                }
            )

    def test_num_results_validation(self):
        """Test num_results validation."""
        # Valid num_results
        request = ChatRequest(
            message="Test message",
            num_results=10,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.num_results == 10

        # num_results too low
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Test message",
                num_results=2,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
                ranking_weights=None,
            )

        # num_results too high
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Test message",
                num_results=251,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
                ranking_weights=None,
            )

    def test_ranking_weights_validation(self):
        """Test ranking_weights validation."""
        # Valid ranking_weights
        valid_weights = {
            "recency_weight": 0.3,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.3,
        }
        request = ChatRequest(
            message="Test message",
            ranking_weights=valid_weights,
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
        )
        assert request.ranking_weights == valid_weights

        # Invalid keys
        invalid_keys = {
            "recency_weight": 0.3,
            "invalid_key": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.3,
        }
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Test message",
                ranking_weights=invalid_keys,
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
            )

        # Invalid values (not between 0 and 1)
        invalid_values = {
            "recency_weight": 1.3,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.3,
        }
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Test message",
                ranking_weights=invalid_values,
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
            )

        # Sum not equal to 1
        invalid_sum = {
            "recency_weight": 0.6,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.3,
        }
        with pytest.raises(ValidationError):
            ChatRequest(
                message="Test message",
                ranking_weights=invalid_sum,
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
            )

    def test_model_validation(self):
        """Test model validation."""
        # Valid model
        request = ChatRequest(
            message="Test message",
            model=OpenAIModel.GPT_4,
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.model == OpenAIModel.GPT_4

        # None is valid
        request = ChatRequest(
            message="Test message",
            model=None,
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.model is None

        # Test invalid model
        with pytest.raises(ValidationError):
            ChatRequest.model_validate(
                {
                    "message": "Test message",
                    "model": "invalid-model",
                    "num_results": 25,
                    "query_type": None,
                    "verbose": False,
                    "generate_alternative_viewpoint": False,
                    "ranking_weights": None,
                }
            )

    def test_max_message_length(self):
        """Test that message can be up to 50,000 characters."""
        long_message = "x" * 50000
        request = ChatRequest(
            message=long_message,
            num_results=25,
            query_type=None,
            verbose=False,
            generate_alternative_viewpoint=False,
            ranking_weights=None,
        )
        assert request.message == long_message

        too_long_message = "x" * 50001
        with pytest.raises(ValidationError):
            ChatRequest(
                message=too_long_message,
                num_results=25,
                query_type=None,
                verbose=False,
                generate_alternative_viewpoint=False,
                ranking_weights=None,
            )


class TestChatResponse:
    """Test suite for ChatResponse schema."""

    def test_required_fields(self):
        """Test that required fields must be present."""
        # Just the minimum required fields
        response = ChatResponse(
            request_id="test-id",
            response="Test response",
            alternative_viewpoints=None,
        )
        assert response.request_id == "test-id"
        assert response.response == "Test response"
        assert response.sources == []
        assert response.processing_time is None
        assert response.alternative_viewpoints is None
        assert isinstance(response.metadata, dict)
        assert len(response.metadata) == 0

    def test_all_fields(self):
        """Test with all fields specified."""
        timestamp = datetime.now()
        metadata = {"key": "value", "model_used": "gpt-4"}
        response = ChatResponse(
            request_id="test-id",
            response="Test response",
            sources=["url1", "url2"],
            processing_time=1.5,
            timestamp=timestamp,
            alternative_viewpoints="Another perspective",
            metadata=metadata,
        )

        assert response.request_id == "test-id"
        assert response.response == "Test response"
        assert response.sources == ["url1", "url2"]
        assert response.processing_time == 1.5
        assert response.timestamp == timestamp
        assert response.alternative_viewpoints == "Another perspective"
        assert response.metadata == metadata

    def test_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now()
        response = ChatResponse(
            request_id="test-id",
            response="Test response",
            alternative_viewpoints=None,
        )
        after = datetime.now()

        assert before <= response.timestamp <= after


class TestHealthStatus:
    """Test suite for HealthStatus schema."""

    def test_health_status_fields(self):
        """Test the HealthStatus fields."""
        status = HealthStatus(
            status="healthy",
            knowledge_base_loaded=True,
            is_loading=False,
        )

        assert status.status == "healthy"
        assert status.knowledge_base_loaded is True
        assert status.is_loading is False

    def test_alternate_values(self):
        """Test with different values."""
        status = HealthStatus(
            status="initializing",
            knowledge_base_loaded=False,
            is_loading=True,
        )

        assert status.status == "initializing"
        assert status.knowledge_base_loaded is False
        assert status.is_loading is True


class TestRootResponse:
    """Test suite for RootResponse schema."""

    def test_root_response_fields(self):
        """Test the RootResponse fields."""
        response = RootResponse(
            message="Welcome to the API",
            status="ok",
        )

        assert response.message == "Welcome to the API"
        assert response.status == "ok"


class TestOpenAIModel:
    """Test OpenAIModel enum and validation methods."""

    def test_is_valid(self):
        """Test the is_valid method."""
        assert OpenAIModel.is_valid("gpt-3.5-turbo") is True
        assert OpenAIModel.is_valid("gpt-4") is True
        assert OpenAIModel.is_valid("gpt-4-turbo") is True
        assert OpenAIModel.is_valid("gpt-100") is False
        assert OpenAIModel.is_valid("") is False
        # Fix the type error by using type casting
        assert OpenAIModel.is_valid(cast(str, None)) is False

    def test_get_default(self):
        """Test the get_default method."""
        assert OpenAIModel.get_default() == "gpt-3.5-turbo"
        assert OpenAIModel.get_default() == OpenAIModel.GPT_3_5_TURBO.value

    def test_enum_values(self):
        """Test that enum values match expected strings."""
        assert OpenAIModel.GPT_3_5_TURBO == "gpt-3.5-turbo"
        assert OpenAIModel.GPT_4 == "gpt-4"
        assert OpenAIModel.GPT_4_TURBO == "gpt-4-turbo"
        assert len(OpenAIModel.__members__) == 3


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

    def test_with_string_value(self):
        """Test instantiation with string value."""
        request = ModelSelectionRequest.model_validate({"model": "gpt-4"})
        assert request.model == "gpt-4"

        with pytest.raises(ValidationError):
            ModelSelectionRequest.model_validate({"model": "invalid-model"})


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
