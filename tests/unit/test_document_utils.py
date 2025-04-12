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

    def test_extract_sources_with_malformed_urls(self):
        """Test extracting sources with malformed URLs."""
        # Create test documents with malformed URLs
        docs = [
            Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}),
            Document(page_content="Doc 2", metadata={"url": "invalid-url-format"}),  # Malformed
            Document(page_content="Doc 3", metadata={"url": "   "}),  # Just whitespace
            Document(page_content="Doc 4", metadata={"url": "javascript:alert(1)"}),  # JavaScript
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions - all URLs should be extracted regardless of format
        # The extract_sources function doesn't validate URL format
        assert len(sources) == 4
        assert "https://example.com/1" in sources
        assert "invalid-url-format" in sources
        assert "   " in sources
        assert "javascript:alert(1)" in sources

    def test_extract_sources_with_url_filtering(self):
        """Test that extract_sources doesn't filter URLs based on content."""
        # Create test documents with various URL types
        docs = [
            Document(
                page_content="Doc 1",
                metadata={"url": "https://example.com/path?query=value#fragment"},
            ),
            Document(page_content="Doc 2", metadata={"url": "ftp://example.com/file.txt"}),
            Document(page_content="Doc 3", metadata={"url": "http://192.168.1.1:8080/admin"}),
            Document(page_content="Doc 4", metadata={"url": "mailto:user@example.com"}),
        ]

        # Extract sources
        sources = extract_sources(docs)

        # Assertions - all URLs should be extracted regardless of scheme
        assert len(sources) == 4
        assert "https://example.com/path?query=value#fragment" in sources
        assert "ftp://example.com/file.txt" in sources
        assert "http://192.168.1.1:8080/admin" in sources
        assert "mailto:user@example.com" in sources


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

    @pytest.mark.asyncio
    async def test_analyze_sentiment_malformed_document(self):
        """Test sentiment analysis with malformed document metadata."""
        # Create test documents with malformed metadata
        docs = [
            Document(page_content="Test content", metadata={}),  # Empty metadata
            Document(page_content=""),  # Empty content
            Document(page_content=""),  # Another empty content
        ]

        # Analyze sentiment
        scored_docs = await analyze_sentiment(docs)

        # Assertions
        assert len(scored_docs) == 3
        # All documents should get neutral sentiment (0.0) when content is missing/empty
        for doc, score in scored_docs:
            assert score == 0.0


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

    @pytest.fixture
    def large_mock_docs_with_timestamps(self) -> List[Tuple[Document, float]]:
        """Create a larger set of mock documents with timestamps for testing large k case."""
        docs = []
        base_ts = 1620000000.0
        for i in range(20):
            sentiment = "bullish" if i % 3 == 0 else ("bearish" if i % 3 == 1 else "neutral")
            doc = Document(page_content=f"{sentiment} document {i}", metadata={"url": f"url{i}"})
            # Vary timestamps
            timestamp = base_ts + (i * 100)
            docs.append((doc, timestamp))
        return docs

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
        """Test document diversification with large k value."""
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
            # Diversify documents with k=6 (larger than the actual docs)
            result = await diversify_documents(mock_docs_with_timestamps, k=6)

            # Assertions
            assert len(result) == 3  # Should return all 3 docs since that's all we have
            # The result should contain all original documents
            assert set(doc.page_content for doc in result) == {
                "Bullish outlook",
                "Bearish outlook",
                "Neutral outlook",
            }

    @pytest.mark.asyncio
    async def test_diversify_documents_large_k_with_many_docs(
        self, large_mock_docs_with_timestamps
    ):
        """Test document diversification with large k value and many documents."""
        # Generate sentiment scores for all docs
        mock_sentiment_results = []
        for i, (doc, _) in enumerate(large_mock_docs_with_timestamps):
            sentiment = (
                0.8
                if "bullish" in doc.page_content
                else (-0.7 if "bearish" in doc.page_content else 0.0)
            )
            mock_sentiment_results.append((doc, sentiment))

        # Create patch for analyze_sentiment
        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Diversify documents with k=10
            result = await diversify_documents(large_mock_docs_with_timestamps, k=10)

            # Assertions
            assert len(result) == 10

            # Check for timestamp range normalization
            earliest_doc = min(large_mock_docs_with_timestamps, key=lambda x: x[1])
            latest_doc = max(large_mock_docs_with_timestamps, key=lambda x: x[1])

            # Latest document should be included due to high recency score
            assert latest_doc[0].page_content in [doc.page_content for doc in result]

            # Check sentiment diversity (should have both positive and negative)
            sentiments = ["bullish", "bearish", "neutral"]
            for sentiment in sentiments:
                matches = [doc for doc in result if sentiment in doc.page_content]
                assert len(matches) > 0, f"No {sentiment} documents in the result"

    @pytest.mark.asyncio
    async def test_diversify_documents_large_k_edge_case(self):
        """Test the large k case with edge conditions like same timestamp."""
        # Create documents with identical timestamps
        docs_with_same_ts = [
            (Document(page_content=f"Doc {i}", metadata={"url": f"url{i}"}), 1620000000.0)
            for i in range(5)
        ]

        # Generate mock sentiment scores
        mock_sentiment_results = [
            (docs_with_same_ts[0][0], 0.8),
            (docs_with_same_ts[1][0], 0.7),
            (docs_with_same_ts[2][0], -0.6),
            (docs_with_same_ts[3][0], -0.5),
            (docs_with_same_ts[4][0], 0.0),
        ]

        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Test with k=3
            result = await diversify_documents(docs_with_same_ts, k=3)

            # Assertions
            assert len(result) == 3

            # Since timestamps are identical, selection should be primarily based on sentiment diversity
            # We should have at least one positive and one negative document
            sentiment_groups = {
                "positive": [0, 1],  # indices of positive sentiment docs
                "negative": [2, 3],  # indices of negative sentiment docs
                "neutral": [4],  # indices of neutral sentiment docs
            }

            result_content_set = set(doc.page_content for doc in result)

            # Check we have at least one positive sentiment doc
            assert any(f"Doc {i}" in result_content_set for i in sentiment_groups["positive"])

            # Check we have at least one negative sentiment doc
            assert any(f"Doc {i}" in result_content_set for i in sentiment_groups["negative"])

    @pytest.mark.asyncio
    async def test_diversify_documents_empty_list(self):
        """Test diversifying an empty document list."""
        result = await diversify_documents([], k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_diversify_documents_k_zero(self, mock_docs_with_timestamps):
        """Test diversification with k=0."""
        result = await diversify_documents(mock_docs_with_timestamps, k=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_diversify_documents_same_sentiments(self, mock_docs_with_timestamps):
        """Test diversification when all documents have the same sentiment."""
        # Mock sentiment scores - all positive
        mock_sentiment_results = [
            (mock_docs_with_timestamps[0][0], 0.8),
            (mock_docs_with_timestamps[1][0], 0.7),
            (mock_docs_with_timestamps[2][0], 0.9),
        ]

        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            result = await diversify_documents(mock_docs_with_timestamps, k=2)
            assert len(result) == 2
            # The newest documents should be prioritized
            assert result[0].page_content == "Neutral outlook"

    @pytest.mark.asyncio
    async def test_diversify_documents_mismatch_docs(self, mock_docs_with_timestamps):
        """Test handling of mismatched documents between sentiment analysis and input."""
        # Create mismatched docs for sentiment analysis
        different_docs = [
            Document(page_content="Different doc 1", metadata={"url": "url1"}),
            Document(page_content="Different doc 2", metadata={"url": "url2"}),
            Document(page_content="Different doc 3", metadata={"url": "url3"}),
        ]

        mock_sentiment_results = [
            (different_docs[0], 0.8),
            (different_docs[1], -0.7),
            (different_docs[2], 0.0),
        ]

        with patch(
            "app.utils.document.analyze_sentiment", AsyncMock(return_value=mock_sentiment_results)
        ):
            # Since doc contents won't match, the algorithm should still handle it gracefully
            result = await diversify_documents(mock_docs_with_timestamps, k=3)

            # In this case, the result should be empty since no documents match
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_diversify_documents_exception_handling(self, mock_docs_with_timestamps):
        """Test exception handling within the function."""
        # Make analyze_sentiment raise an exception
        with patch(
            "app.utils.document.analyze_sentiment",
            AsyncMock(side_effect=Exception("Test exception")),
        ):
            # Function should handle the exception and return empty list
            with pytest.raises(Exception):
                await diversify_documents(mock_docs_with_timestamps, k=3)

    @pytest.mark.asyncio
    async def test_diversify_documents_analyze_sentiment_error_propagation(self):
        """Test that diversify_documents properly propagates errors from analyze_sentiment."""
        # Create sample documents
        docs_with_timestamps = [
            (Document(page_content="Test doc 1", metadata={"url": "url1"}), 1620000000.0),
            (Document(page_content="Test doc 2", metadata={"url": "url2"}), 1620000100.0),
        ]

        # Make analyze_sentiment raise a specific exception type we can check
        class TestAnalysisError(Exception):
            """Test-specific error for sentiment analysis."""

            pass

        # Create patch to simulate analyze_sentiment failure
        with patch(
            "app.utils.document.analyze_sentiment",
            AsyncMock(side_effect=TestAnalysisError("Test analysis failure")),
        ):
            # Test that the exception type is preserved
            with pytest.raises(TestAnalysisError):
                await diversify_documents(docs_with_timestamps, k=2)

            # The exception message should match
            try:
                await diversify_documents(docs_with_timestamps, k=2)
            except TestAnalysisError as e:
                assert str(e) == "Test analysis failure"
