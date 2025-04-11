"""
Schema definitions for OpenAI model selection.

This module defines the available OpenAI models that can be used for generation,
along with validation logic and configuration for model selection in API requests.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class OpenAIModel(str, Enum):
    """
    Enum of allowed OpenAI models for generation.

    These models represent the available options for text generation.
    Different models have different capabilities and pricing.
    """

    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"

    @classmethod
    def is_valid(cls, model: str) -> bool:
        """
        Check if a model string is valid.

        Args:
            model (str): The model string to validate

        Returns:
            bool: True if the model is valid, False otherwise
        """
        return model in [item.value for item in cls]

    @classmethod
    def get_default(cls):
        """
        Get the default model.

        Returns:
            str: The default model value
        """
        return cls.GPT_3_5_TURBO.value


class ModelSelectionRequest(BaseModel):
    """
    Schema for model selection in requests.

    This model can be used as a parameter in API endpoints to specify
    which language model should be used for the generation.
    """

    model: Optional[OpenAIModel] = Field(
        default=None,
        description="The OpenAI model to use for generating the response. If not specified, the default model will be used.",
    )

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True  # Use the string values of the enum
