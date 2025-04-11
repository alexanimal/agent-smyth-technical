"""
Document processing utilities for the RAG system.

This module provides utility functions for working with documents in the
Retrieval Augmented Generation (RAG) system, including source extraction,
sentiment analysis, and diversity-aware document re-ranking.
"""

from typing import List, Set, Tuple

from langchain_core.documents import Document


def extract_sources(documents: List[Document]) -> List[str]:
    """
    Extract source URLs from documents.

    Args:
        documents: List of Document objects with metadata

    Returns:
        List[str]: List of unique source URLs from the documents
    """
    sources: Set[str] = set()

    for doc in documents:
        if doc.metadata and "url" in doc.metadata and doc.metadata["url"]:
            sources.add(doc.metadata["url"])

    return list(sources)


async def analyze_sentiment(documents: List[Document]) -> List[Tuple[Document, float]]:
    """
    Analyze sentiment of documents to enable diversity-aware re-ranking.

    This simplified sentiment analysis assigns sentiment scores to documents
    to help ensure representation of diverse viewpoints in the final context.

    Args:
        documents: List of Document objects to analyze

    Returns:
        List of tuples containing (document, sentiment_score)
        where sentiment_score ranges from -1.0 (very negative) to 1.0 (very positive)
    """
    scored_docs = []

    for doc in documents:
        # Simple keyword-based sentiment analysis
        # In a production system, this would use a proper sentiment analysis model
        content = doc.page_content.lower()

        # Positive sentiment keywords
        pos_words = [
            "bullish",
            "positive",
            "uptrend",
            "buy",
            "growth",
            "oversold",
            "support",
            "opportunity",
            "undervalued",
            "breakout",
            "upside",
        ]

        # Negative sentiment keywords
        neg_words = [
            "bearish",
            "negative",
            "downtrend",
            "sell",
            "decline",
            "overbought",
            "resistance",
            "risk",
            "overvalued",
            "breakdown",
            "downside",
        ]

        # Count keyword occurrences
        pos_count = sum(content.count(word) for word in pos_words)
        neg_count = sum(content.count(word) for word in neg_words)

        # Calculate sentiment score (-1 to 1)
        total = pos_count + neg_count
        if total > 0:
            sentiment = (pos_count - neg_count) / total
        else:
            sentiment = 0.0  # Neutral if no sentiment words found

        scored_docs.append((doc, sentiment))

    return scored_docs


async def diversify_documents(
    docs_with_timestamps: List[Tuple[Document, float]], k: int
) -> List[Document]:
    """
    Re-rank documents to balance recency and viewpoint diversity.

    This method ensures the final selection of documents presents diverse
    perspectives by balancing recency with sentiment diversity.

    Args:
        docs_with_timestamps: Documents with timestamp information
        k: Number of documents to select

    Returns:
        List of selected documents with diverse viewpoints
    """
    if not docs_with_timestamps or k <= 0:
        return []

    # Extract sentiment scores
    docs_with_sentiment = await analyze_sentiment([doc for doc, _ in docs_with_timestamps])

    # Combine timestamp and sentiment information
    docs_with_metadata = []
    for (doc1, timestamp), (doc2, sentiment) in zip(docs_with_timestamps, docs_with_sentiment):
        # Verify they're the same document
        if doc1.page_content == doc2.page_content:
            docs_with_metadata.append((doc1, timestamp, sentiment))

    # If k is small, ensure diversity directly
    if k <= 5:
        # First, sort by recency (highest timestamp first)
        docs_with_metadata.sort(key=lambda x: x[1], reverse=True)

        # Take top k/2 most recent documents (rounded up)
        recent_half = docs_with_metadata[: max(1, k // 2)]

        # For the rest, ensure diverse sentiment
        remaining = docs_with_metadata[max(1, k // 2) :]

        # Sort remaining by absolute sentiment (to get strongest opinions)
        remaining.sort(key=lambda x: abs(x[2]), reverse=True)

        # Group by positive and negative sentiment
        positive = [item for item in remaining if item[2] > 0]
        negative = [item for item in remaining if item[2] < 0]
        neutral = [item for item in remaining if item[2] == 0]

        # Alternate selection from positive and negative groups
        diverse_selection: List[Tuple[Document, float, float]] = []
        while len(diverse_selection) < (k - len(recent_half)) and (positive or negative or neutral):
            if positive and (not diverse_selection or diverse_selection[-1][2] <= 0):
                diverse_selection.append(positive.pop(0))
            elif negative and (not diverse_selection or diverse_selection[-1][2] >= 0):
                diverse_selection.append(negative.pop(0))
            elif neutral:
                diverse_selection.append(neutral.pop(0))
            elif positive:
                diverse_selection.append(positive.pop(0))
            elif negative:
                diverse_selection.append(negative.pop(0))

        # Combine recent and diverse selections
        combined = recent_half + diverse_selection
        return [item[0] for item in combined[:k]]

    # For larger k, use a more sophisticated approach
    else:
        # Get timestamp range for normalization
        if docs_with_metadata:
            max_ts = max(item[1] for item in docs_with_metadata)
            min_ts = min(item[1] for item in docs_with_metadata)
            ts_range = max(0.001, max_ts - min_ts)  # Avoid division by zero

            # Compute combined score = 0.7*recency + 0.3*diversity
            scored_docs = []

            # Track sentiments we've seen so far
            selected_sentiments: List[float] = []

            for doc, ts, sentiment in docs_with_metadata:
                # Normalize timestamp to 0-1 (1 = most recent)
                recency_score = (ts - min_ts) / ts_range

                # Compute diversity score based on how different this document's
                # sentiment is from already selected documents
                diversity_score = 1.0
                if selected_sentiments:
                    # Lower score if similar sentiments already selected
                    similarity = min(abs(sentiment - s) for s in selected_sentiments)
                    diversity_score = min(1.0, similarity * 5)  # Scale up for effect

                # Combined score with weightings
                combined_score = (0.7 * recency_score) + (0.3 * diversity_score)
                scored_docs.append((doc, combined_score, sentiment))

            # Sort by combined score (highest first)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Select top documents
            result = []
            for i in range(min(k, len(scored_docs))):
                result.append(scored_docs[i][0])
                selected_sentiments.append(scored_docs[i][2])

            return result

        return [doc for doc, _, _ in docs_with_metadata[:k]]
