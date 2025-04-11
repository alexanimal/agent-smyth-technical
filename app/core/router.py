"""
Query classification router for the chat system.

This module contains the ChatRouter class, which is responsible for classifying
incoming queries using an LLM to determine the most appropriate processing strategy.
"""

import json
import logging
from operator import itemgetter
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.prompts import PromptManager

# Configure logging
logger = logging.getLogger(__name__)


class ChatRouter:
    """
    Routes chat queries to different handlers based on the content with confidence scoring.

    This class is responsible for analyzing query content and determining
    the most appropriate query type and processing strategy to use.
    It provides confidence scores for multiple categories to enable
    more nuanced handling of mixed-type queries.

    Attributes:
        classifier_model: The LLM used for query classification
    """

    def __init__(self, classifier_model: BaseChatModel):
        """
        Initialize the ChatRouter with a classifier model.

        Args:
            classifier_model: The LLM to use for query classification
        """
        self.classifier_model = classifier_model

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classifies a query and returns confidence scores for multiple categories.

        Uses an LLM to analyze the query content and provide confidence scores for
        each possible category, allowing for nuanced, non-binary classification.

        Args:
            query: The user's query

        Returns:
            Dict containing:
                - query_type: The highest confidence category
                - confidence_scores: Dict mapping categories to confidence values (0-100)
                - is_mixed: Boolean indicating if query spans multiple categories significantly

        Raises:
            None: Defaults to "general" with low confidence if classification fails
        """
        # Get the classification prompt from PromptManager
        classification_prompt = PromptManager.get_classification_prompt()

        # Create a chain for classification
        chain = classification_prompt | self.classifier_model | StrOutputParser()

        # Get the classification
        try:
            classification_result = await chain.ainvoke({"query": query})

            # Parse the JSON result with confidence scores
            confidence_scores = json.loads(classification_result.strip())

            # Validate the format
            expected_keys = ["technical", "trading_thesis", "investment", "general"]
            if not all(key in confidence_scores for key in expected_keys):
                logger.warning(
                    f"Invalid classification result: {classification_result}, defaulting to general"
                )
                confidence_scores = {
                    "technical": 0,
                    "trading_thesis": 0,
                    "investment": 0,
                    "general": 100,
                }

            # Determine the highest confidence category
            max_category = max(confidence_scores.items(), key=itemgetter(1))[0]

            # Determine if this is a mixed query (second highest score > threshold)
            scores = sorted(confidence_scores.values(), reverse=True)
            is_mixed = len(scores) > 1 and scores[1] > 30  # Second score > 30% indicates mixed type

            result = {
                "query_type": max_category,
                "confidence_scores": confidence_scores,
                "is_mixed": is_mixed,
            }

            logger.info(
                f"Query classified as: {max_category} with confidence scores: {confidence_scores}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in classification: {str(e)}", exc_info=True)
            # Default to general with low confidence
            return {
                "query_type": "general",
                "confidence_scores": {
                    "technical": 0,
                    "trading_thesis": 0,
                    "investment": 0,
                    "general": 100,
                },
                "is_mixed": False,
            }
