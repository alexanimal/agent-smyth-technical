"""
Utility functions for RAG components.

This module provides utilities for common operations in the RAG workflow,
including caching and optimization for LLM instances.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_cached_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Get a cached LLM instance for better performance.

    This function creates and caches LLM instances to avoid redundant initialization,
    which can improve performance when the same model is used multiple times.

    Args:
        model_name: Name of the OpenAI model to use
        temperature: Temperature parameter for generation (0-1)

    Returns:
        A cached ChatOpenAI instance

    Raises:
        ValueError: If the model name is not in the allowed list
    """
    if model_name not in settings.allowed_models:
        logger.warning(
            f"Requested model '{model_name}' not in allowed models, falling back to {settings.default_model}"
        )
        model_name = settings.default_model

    logger.debug(f"Creating or retrieving cached LLM instance for model: {model_name}")
    return ChatOpenAI(model_name=model_name, temperature=temperature)  # type: ignore


async def generate_with_fallback(
    prompt: Any, model_name: str, fallback_model: str = "gpt-3.5-turbo", temperature: float = 0.0
) -> Any:
    """
    Generate a response with automatic fallback to a simpler model if the primary model fails.

    This function attempts to generate a response with the specified model,
    and falls back to a more reliable model if an error occurs.

    Args:
        prompt: The input prompt to send to the model
        model_name: The name of the primary model to use
        fallback_model: The name of the fallback model to use
        temperature: Temperature parameter for generation

    Returns:
        The generated response

    Raises:
        Exception: If both primary and fallback models fail
    """
    start_time = time.time()
    try:
        model = get_cached_llm(model_name, temperature)
        logger.info(f"Generating with model: {model_name}")
        result = await model.ainvoke(prompt)
        duration = time.time() - start_time
        logger.info(f"Generated with {model_name} in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.warning(
            f"Model {model_name} failed after {duration:.2f}s: {str(e)}. "
            f"Falling back to {fallback_model}"
        )

        try:
            fallback = get_cached_llm(fallback_model, temperature)
            fallback_start = time.time()
            result = await fallback.ainvoke(prompt)
            fallback_duration = time.time() - fallback_start
            logger.info(f"Generated with fallback {fallback_model} in {fallback_duration:.2f}s")
            return result
        except Exception as fallback_error:
            total_duration = time.time() - start_time
            logger.error(
                f"Both primary and fallback models failed. "
                f"Total duration: {total_duration:.2f}s. "
                f"Fallback error: {str(fallback_error)}"
            )
            raise Exception(f"Failed to generate response: {str(fallback_error)}")
