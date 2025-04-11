"""
Document scoring module for the RAG workflow.

This module provides functions to score documents based on various signals,
including relevance, recency, and social engagement metrics.
"""

import logging
import math
from typing import Dict, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to a 0-1 scale.

    Args:
        value: The value to normalize
        min_val: The minimum value in the range
        max_val: The maximum value in the range

    Returns:
        A normalized value between 0 and 1
    """
    if max_val == min_val:
        return 0.5  # Default when all values are the same

    return (value - min_val) / (max_val - min_val)


def logarithmic_scale(value: float, base: float = 10) -> float:
    """
    Apply logarithmic scaling to a value to handle wide ranges.

    Args:
        value: The value to scale
        base: The logarithm base (default: 10)

    Returns:
        The logarithmically scaled value
    """
    if value <= 0:
        return 0
    return math.log(value + 1, base)  # Add 1 to handle values between 0 and 1


def score_documents_with_social_metrics(
    docs: List[Document], weights: Dict[str, float]
) -> List[Tuple[Document, float]]:
    """
    Score documents based on a combination of signals including social metrics.

    Args:
        docs: List of documents to score
        weights: Dictionary of weights for different signals
            - recency_weight: Weight for recency score (timestamp)
            - view_weight: Weight for view count
            - like_weight: Weight for like count
            - retweet_weight: Weight for retweet count

    Returns:
        List of (document, score) tuples, sorted by descending score
    """
    if not docs:
        return []

    # Extract weights with defaults
    recency_weight = weights.get("recency_weight", 0.4)
    view_weight = weights.get("view_weight", 0.2)
    like_weight = weights.get("like_weight", 0.2)
    retweet_weight = weights.get("retweet_weight", 0.2)

    # Collect metrics for normalization
    timestamps = []
    view_counts = []
    like_counts = []
    retweet_counts = []

    for doc in docs:
        metadata = doc.metadata

        # Extract metrics with fallbacks to 0
        timestamp = float(metadata.get("timestamp_unix", 0))
        view_count = int(metadata.get("viewCount", 0))
        like_count = int(metadata.get("likeCount", 0))
        retweet_count = int(metadata.get("retweetCount", 0))

        timestamps.append(timestamp)
        view_counts.append(view_count)
        like_counts.append(like_count)
        retweet_counts.append(retweet_count)

    # Find min/max for normalization
    min_timestamp = min(timestamps) if timestamps else 0
    max_timestamp = max(timestamps) if timestamps else 1

    # Apply logarithmic scaling to engagement metrics to handle viral outliers
    log_view_counts = [logarithmic_scale(count) for count in view_counts]
    log_like_counts = [logarithmic_scale(count) for count in like_counts]
    log_retweet_counts = [logarithmic_scale(count) for count in retweet_counts]

    min_log_view = min(log_view_counts) if log_view_counts else 0
    max_log_view = max(log_view_counts) if log_view_counts else 1
    min_log_like = min(log_like_counts) if log_like_counts else 0
    max_log_like = max(log_like_counts) if log_like_counts else 1
    min_log_retweet = min(log_retweet_counts) if log_retweet_counts else 0
    max_log_retweet = max(log_retweet_counts) if log_retweet_counts else 1

    # Score each document
    scored_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata

        # Extract metrics with fallbacks
        timestamp = float(metadata.get("timestamp_unix", 0))
        view_count = int(metadata.get("viewCount", 0))
        like_count = int(metadata.get("likeCount", 0))
        retweet_count = int(metadata.get("retweetCount", 0))

        # Calculate normalized scores
        recency_score = normalize_value(timestamp, min_timestamp, max_timestamp)

        # Apply logarithmic scaling and normalize engagement metrics
        log_view = logarithmic_scale(view_count)
        log_like = logarithmic_scale(like_count)
        log_retweet = logarithmic_scale(retweet_count)

        view_score = normalize_value(log_view, min_log_view, max_log_view)
        like_score = normalize_value(log_like, min_log_like, max_log_like)
        retweet_score = normalize_value(log_retweet, min_log_retweet, max_log_retweet)

        # Calculate composite score
        composite_score = (
            recency_weight * recency_score
            + view_weight * view_score
            + like_weight * like_score
            + retweet_weight * retweet_score
        )

        # Add document and score to the result list
        scored_docs.append((doc, composite_score))

        # Add detailed scoring to metadata for debugging if needed
        doc.metadata["_scoring"] = {
            "recency_score": recency_score,
            "view_score": view_score,
            "like_score": like_score,
            "retweet_score": retweet_score,
            "composite_score": composite_score,
        }

    # Sort by descending score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs


def diversify_ranked_documents(
    scored_docs: List[Tuple[Document, float]], k: int, diversity_factor: float = 0.3
) -> List[Document]:
    """
    Apply diversification to the top-ranked documents.

    Args:
        scored_docs: List of (document, score) tuples, sorted by score
        k: Number of documents to return
        diversity_factor: Factor controlling diversity (0-1)
            0 means pure ranking, 1 means maximum diversity

    Returns:
        Diversified list of documents
    """
    if not scored_docs or k <= 0:
        return []

    if len(scored_docs) <= k:
        return [doc for doc, _ in scored_docs]

    # Simple diversity implementation:
    # 1. Take the top (1-diversity_factor)*k documents directly
    # 2. For the remaining slots, select documents to maximize diversity

    # Take the top documents directly
    num_top = max(1, int((1 - diversity_factor) * k))
    result = [doc for doc, _ in scored_docs[:num_top]]

    # For remaining slots, implement a basic diversity selection
    remaining = k - num_top
    candidates = [doc for doc, _ in scored_docs[num_top:]]

    # Simple diversification: take documents that increase time range coverage
    # In a real implementation, this would analyze topics, sentiments, etc.
    if candidates and remaining > 0:
        # For this example, just take evenly spaced documents from the rest
        step = max(1, len(candidates) // remaining)
        for i in range(0, len(candidates), step):
            if len(result) < k:
                result.append(candidates[i])

    return result
