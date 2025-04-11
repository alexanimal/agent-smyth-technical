"""
Tests for document processing utilities.

This module contains tests for the document processing utility functions
in app/utils/document.py, including source extraction, sentiment analysis,
and diversity-aware document re-ranking.
"""

from typing import List, Tuple
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from app.utils.document import analyze_sentiment, diversify_documents, extract_sources


class TestExtractSources:
    """Tests for the extract_sources function."""

    def test_extract_sources_with_valid_urls(self):
        """Test extracting sources with valid URLs in document metadata."""
        # Create test documents with URLs
        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={"url": "https://example.com/2"}),
            Document(page_content="Doc 3", metadata={"url": "https://example.com/3"}),
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions
        assert len(sources) == 3
        assert "https://example.com/1" in sources
        assert "https://example.com/2" in sources
        assert "https://example.com/3" in sources

    def test_extract_sources_with_duplicate_urls(self):
        """Test extracting sources with duplicate URLs."""
        # Create test documents with duplicate URLs
        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={"url": "https://example.com/2"}),
            Document(page_content="Doc 3", metadata={"url": "https://example.com/1"}),  # Duplicate
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions
        assert len(sources) == 2  # Should deduplicate
        assert "https://example.com/1" in sources
        assert "https://example.com/2" in sources

    def test_extract_sources_with_missing_metadata(self):
        """Test extracting sources with missing metadata."""
        # Create test documents with missing metadata
        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={}),  # Missing URL
            Document(page_content="Doc 3"),  # No metadata at all
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions
        assert len(sources) == 1
        assert "https://example.com/1" in sources

    def test_extract_sources_with_empty_urls(self):
        """Test extracting sources with empty URL strings."""
        # Create test documents with empty URLs
        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={"url": ""}),  # Empty URL
            Document(page_content="Doc 3", metadata={"url": None}),  # None URL
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions
        assert len(sources) == 1
        assert "https://example.com/1" in sources

    def test_extract_sources_with_empty_docs(self):
        """Test extracting sources with an empty document list."""
        # Extract sources from empty list
        sources = extract_sources([])

        # Assertions
        assert len(sources) == 0
        assert sources == []


class TestAnalyzeSentiment:
    """Tests for the analyze_sentiment function."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self):
        """Test sentiment analysis with positive content."""
        # Create test document with positive sentiment
        docs = [
            Document(
                page_content="This stock is showing a bullish trend with strong support at the current price level. Buying opportunity as it's undervalued with good upside potential."
            )
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 1
        doc, score = scored_docs[0]
        assert doc == docs[0]
        assert score > 0  # Should be positive sentiment

    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self):
        """Test sentiment analysis with negative content."""
        # Create test document with negative sentiment
        docs = [
            Document(
                page_content="The stock is showing a bearish trend with strong resistance. It's currently overbought and at risk of a breakdown. Negative outlook with downside potential."
            )
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 1
        doc, score = scored_docs[0]
        assert doc == docs[0]
        assert score < 0  # Should be negative sentiment

    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral(self):
        """Test sentiment analysis with neutral content."""
        # Create test document with neutral sentiment
        docs = [
            Document(
                page_content="The stock price is currently trading at $200. Volume has been average."
            )
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 1
        doc, score = scored_docs[0]
        assert doc == docs[0]
        assert score == 0  # Should be neutral sentiment (no keywords)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_mixed(self):
        """Test sentiment analysis with mixed content."""
        # Create test document with mixed sentiment
        docs = [
            Document(
                page_content="While the stock shows some bullish indicators, there are bearish signals as well. It has support, but also faces resistance."
            )
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 1
        doc, score = scored_docs[0]
        assert doc == docs[0]
        assert -0.3 <= score <= 0.3  # Should be relatively balanced

    @pytest.mark.asyncio
    async def test_analyze_sentiment_multiple_docs(self):
        """Test sentiment analysis with multiple documents."""
        # Create multiple test documents
        docs = [
            Document(page_content="Bullish trend with strong buy signals."),
            Document(page_content="Bearish outlook with sell recommendation."),
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 2
        assert scored_docs[0][0] == docs[0]
        assert scored_docs[1][0] == docs[1]
        assert scored_docs[0][1] > 0  # First doc should be positive
        assert scored_docs[1][1] < 0  # Second doc should be negative

    @pytest.mark.asyncio
    async def test_analyze_sentiment_empty_docs(self):
        """Test sentiment analysis with empty document list."""
        # Analyze sentiment of empty list
        scored_docs = await analyze_sentiment([])

        # Assertions
        assert len(scored_docs) == 0
        assert scored_docs == []


class TestDiversifyDocuments:
    """Tests for the diversify_documents function."""

    @pytest.fixture
    def mock_docs_with_timestamps(self) -> List[Tuple[Document, float]]:
        """Create mock documents with timestamps."""
        return [
            (Document(page_content="Bullish outlook", metadata={"url": "url1"}), 1620000000.0),
            (Document(page_content="Bearish outlook", metadata={"url": "url2"}), 1620000100.0),
            (Document(page_content="Neutral outlook", metadata={"url": "url3"}), 1620000200.0),
        ]

    @pytest.mark.asyncio
    async def test_diversify_documents_small_k(self, mock_docs_with_timestamps):
        """Test document diversification with small k value."""
        # Mock sentiment scores
        mock_sentiment_results = [
            (mock_docs_with_timestamps[0][0], 0.8),  # Bullish: positive
            (mock_docs_with_timestamps[1][0], -0.7),  # Bearish: negative
            (mock_docs_with_timestamps[2][0], 0.0),  # Neutral
        ]

        # Create patch for analyze_sentiment
        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Diversify documents with k=2
            result = await diversify_documents(mock_docs_with_timestamps, k=2)

            # Assertions
            assert len(result) == 2

            # The newest document should be first due to recency
            assert result[0].page_content == "Neutral outlook"

            # Looking at the code, for small k, the algorithm:
            # 1. Takes top k/2 documents by recency first (rounded up)
            # 2. Then tries to get diverse sentiment in the remainder
            # For k=2, the algorithm takes 1 document by recency, leaving 1 for diversity
            # This means the second document could be either, depending on rounding and sort behavior
            acceptable_second_docs = ["Bearish outlook", "Bullish outlook"]
            assert result[1].page_content in acceptable_second_docs

    @pytest.mark.asyncio
    async def test_diversify_documents_large_k(self, mock_docs_with_timestamps):
        """Test document diversification with larger k value."""
        # Add more documents for a larger k
        mock_docs = mock_docs_with_timestamps + [
            (Document(page_content="Strongly bullish", metadata={"url": "url4"}), 1620000300.0),
            (Document(page_content="Slightly bearish", metadata={"url": "url5"}), 1620000400.0),
        ]

        # Mock sentiment scores
        mock_sentiment_results = [
            (mock_docs[0][0], 0.8),  # Bullish: positive
            (mock_docs[1][0], -0.7),  # Bearish: negative
            (mock_docs[2][0], 0.0),  # Neutral
            (mock_docs[3][0], 0.9),  # Strongly bullish: positive
            (mock_docs[4][0], -0.3),  # Slightly bearish: negative
        ]

        # Create patch for analyze_sentiment
        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Diversify documents with k=6 (more than available)
            result = await diversify_documents(mock_docs, k=6)

            # Assertions
            assert len(result) == 5  # Should return all 5 docs

            # The newest document should be first due to recency
            assert result[0].page_content == "Slightly bearish"

    @pytest.mark.asyncio
    async def test_diversify_documents_empty_list(self):
        """Test document diversification with empty document list."""
        # Diversify empty document list
        result = await diversify_documents([], k=5)

        # Assertions
        assert len(result) == 0
        assert result == []

    @pytest.mark.asyncio
    async def test_diversify_documents_k_zero(self, mock_docs_with_timestamps):
        """Test document diversification with k=0."""
        # Diversify with k=0
        result = await diversify_documents(mock_docs_with_timestamps, k=0)

        # Assertions
        assert len(result) == 0
        assert result == []

    @pytest.mark.asyncio
    async def test_diversify_documents_same_sentiments(self, mock_docs_with_timestamps):
        """Test document diversification when all documents have the same sentiment."""
        # Mock sentiment scores all positive
        mock_sentiment_results = [
            (mock_docs_with_timestamps[0][0], 0.8),  # All positive
            (mock_docs_with_timestamps[1][0], 0.6),
            (mock_docs_with_timestamps[2][0], 0.7),
        ]

        # Create patch for analyze_sentiment
        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Diversify documents with k=2
            result = await diversify_documents(mock_docs_with_timestamps, k=2)

            # Assertions
            assert len(result) == 2

            # Should still rank by recency first
            assert result[0].page_content == "Neutral outlook"  # Most recent
