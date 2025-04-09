"""
Unit tests for schema classes and enums.
"""

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, HealthStatus, QueryType, RootResponse


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
        )

        assert request.num_results == 5
        assert request.verbose is False
        # assert request.context == {}
        # assert request.model == "gpt-4o"
        assert request.query_type is None

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
        )
        assert request.query_type is None

        # Invalid query type should raise ValidationError - using a non-QueryType value that will trigger validation
        # We need to test this at runtime since the type checker prevents using string literals directly
        invalid_value = "invalid_type"
        with pytest.raises(ValidationError):
            # Use a dict to bypass static type checking
            ChatRequest.parse_obj(
                {
                    "message": "Test message",
                    "num_results": 5,
                    "verbose": False,
                    "context": {},
                    "model": "gpt-4o",
                    "query_type": invalid_value,
                }
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
