"""
Unit tests for the PromptManager class in app.prompts
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

from app.prompts import PromptManager


def assert_message_contains(message, text: str):
    """
    Assert that a message template contains specific text.
    Works with different LangChain versions by checking multiple possible locations.
    """
    # Try different ways to access template content based on API version
    content = None

    # Try prompt.template
    if hasattr(message, "prompt") and hasattr(message.prompt, "template"):
        content = message.prompt.template

    # Try direct content attribute
    elif hasattr(message, "content"):
        content = message.content

    # Try template attribute
    elif hasattr(message, "template"):
        content = message.template

    assert content is not None, f"Cannot find content in message: {message}"
    assert text in content, f"Expected '{text}' in: {content}"


class TestPromptManager:
    """Test suite for the PromptManager class."""

    def test_get_investment_prompt_structure(self):
        """Test the structure of the investment prompt template."""
        # Act
        prompt = PromptManager.get_investment_prompt()

        # Assert
        assert isinstance(prompt, ChatPromptTemplate)
        messages = prompt.messages

        # Check we have the expected number of messages
        assert len(messages) == 3

        # Use our updated assertion helper
        assert_message_contains(messages[0], "financial advisor")
        assert_message_contains(messages[0], "Long")
        assert_message_contains(messages[0], "Short")
        assert_message_contains(messages[0], "Hold")

        # Check input variables
        input_variables = prompt.input_variables
        assert "question" in input_variables
        assert "context" in input_variables

    def test_get_general_prompt_structure(self):
        """Test the structure of the general prompt template."""
        # Act
        prompt = PromptManager.get_general_prompt()

        # Assert
        assert isinstance(prompt, ChatPromptTemplate)
        messages = prompt.messages

        # Check message placeholders
        placeholder_found = False
        for message in messages:
            if isinstance(message, MessagesPlaceholder):
                placeholder_found = True
                assert message.variable_name == "chat_history"
                assert message.optional is True
        assert placeholder_found, "Chat history placeholder not found"

        # Check input variables
        input_variables = prompt.input_variables
        assert "question" in input_variables
        assert "context" in input_variables

    def test_get_trading_thesis_prompt_structure(self):
        """Test the structure of the trading thesis prompt template."""
        # Act
        prompt = PromptManager.get_trading_thesis_prompt()

        # Assert
        assert isinstance(prompt, ChatPromptTemplate)
        messages = prompt.messages

        # Check we have the expected number of messages
        assert len(messages) == 3

        # Check for key sections in the system message using our helper
        key_sections = [
            "ANALYSIS PHASE",
            "THESIS DEVELOPMENT",
            "BIAS MITIGATION",
            "SUPPORTING EVIDENCE",
            "FORMAT",
        ]
        for section in key_sections:
            assert_message_contains(messages[0], section)

        # Check input variables
        input_variables = prompt.input_variables
        assert "question" in input_variables
        assert "context" in input_variables

    def test_get_classification_prompt_structure(self):
        """Test the structure of the classification prompt template."""
        # Act
        prompt = PromptManager.get_classification_prompt()

        # Assert
        assert isinstance(prompt, ChatPromptTemplate)
        messages = prompt.messages

        # Check we have the expected number of messages
        assert len(messages) == 2

        # Check message types
        assert isinstance(messages[0], SystemMessagePromptTemplate)
        assert isinstance(messages[1], HumanMessagePromptTemplate)

        # Check key content using our helper
        assert_message_contains(messages[0], "query classifier")
        assert_message_contains(messages[0], "investment")
        assert_message_contains(messages[0], "trading_thesis")
        assert_message_contains(messages[0], "general")

        # Check input variables
        input_variables = prompt.input_variables
        assert "query" in input_variables
        assert len(input_variables) == 1  # Should only have one input variable

    def test_prompt_format_with_inputs(self):
        """Test that prompts format correctly with inputs."""
        # Arrange
        test_question = "What's the sentiment on Tesla stock?"
        test_context = "Tweet 1: Tesla stock looks promising.\nTweet 2: I'm bullish on Tesla."

        # Act - Test all three prompt types
        investment_prompt = PromptManager.get_investment_prompt()
        general_prompt = PromptManager.get_general_prompt()
        thesis_prompt = PromptManager.get_trading_thesis_prompt()

        # Format the prompts with inputs
        investment_formatted = investment_prompt.format_messages(
            question=test_question, context=test_context
        )
        general_formatted = general_prompt.format_messages(
            question=test_question, context=test_context
        )
        thesis_formatted = thesis_prompt.format_messages(
            question=test_question, context=test_context
        )

        # Assert - Check that inputs are correctly inserted
        # For investment prompt
        assert any(test_question in msg.content for msg in investment_formatted)
        assert any(test_context in msg.content for msg in investment_formatted)

        # For general prompt
        assert any(test_question in msg.content for msg in general_formatted)
        assert any(test_context in msg.content for msg in general_formatted)

        # For thesis prompt
        assert any(test_question in msg.content for msg in thesis_formatted)
        assert any(test_context in msg.content for msg in thesis_formatted)

    def test_classification_prompt_format(self):
        """Test that the classification prompt formats correctly with inputs."""
        # Arrange
        test_query = "Should I buy Tesla stock based on recent tweets?"

        # Act
        classification_prompt = PromptManager.get_classification_prompt()
        formatted = classification_prompt.format_messages(query=test_query)

        # Assert
        assert len(formatted) == 2
        assert formatted[1].content == test_query

    def test_prompt_with_chat_history(self):
        """Test that the general prompt correctly handles chat history."""
        # Arrange
        test_question = "Follow-up question about Tesla"
        test_context = "Some context about Tesla"
        test_chat_history = [
            {"type": "human", "content": "What do you think about Tesla?"},
            {"type": "ai", "content": "Based on the tweets, sentiment is positive."},
        ]

        # Act
        prompt = PromptManager.get_general_prompt()
        formatted = prompt.format_messages(
            question=test_question, context=test_context, chat_history=test_chat_history
        )

        # Assert
        # Check chat history is included in the formatted messages
        # Ensure only string content is joined
        all_content = " ".join(
            [
                msg.content
                for msg in formatted
                if hasattr(msg, "content") and isinstance(msg.content, str)
            ]
        )
        assert "What do you think about Tesla?" in all_content
        assert "Based on the tweets, sentiment is positive." in all_content

    @patch("langchain_core.prompts.ChatPromptTemplate.from_messages")
    def test_prompt_creation(self, mock_from_messages):
        """Test that prompts are created using the correct method."""
        # Arrange
        mock_template = MagicMock()
        mock_from_messages.return_value = mock_template

        # Act
        result = PromptManager.get_investment_prompt()

        # Assert
        assert mock_from_messages.called
        assert result == mock_template


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
