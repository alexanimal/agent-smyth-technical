"""
Tests for the document scoring module.

This module contains tests for the scoring utility functions used in the
Retrieval Augmented Generation workflow, including document normalization,
scaling, and scoring based on various signals.
"""

import math
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.scoring import (
    diversify_ranked_documents,
    logarithmic_scale,
    normalize_value,
    score_documents_with_social_metrics,
)


class TestNormalizeValue:
    """Tests for the normalize_value function."""

    def test_normalize_basic(self):
        """Test basic normalization with standard inputs."""
        # Test with values within range
        assert normalize_value(5, 0, 10) == 0.5
        assert normalize_value(0, 0, 10) == 0.0
        assert normalize_value(10, 0, 10) == 1.0
        assert normalize_value(7.5, 0, 10) == 0.75

    def test_normalize_equal_min_max(self):
        """Test normalization when min and max are equal."""
        # When min=max, function should return 0.5 to prevent division by zero
        assert normalize_value(5, 5, 5) == 0.5
        assert normalize_value(0, 0, 0) == 0.5

    def test_normalize_out_of_range(self):
        """Test normalization with values outside the min-max range."""
        # Values below min
        assert normalize_value(-5, 0, 10) == -0.5

        # Values above max
        assert normalize_value(15, 0, 10) == 1.5

    def test_normalize_negative_range(self):
        """Test normalization with negative ranges."""
        # Negative min and max
        assert normalize_value(-5, -10, -5) == 1.0
        assert normalize_value(-7.5, -10, -5) == 0.5

        # Negative to positive range
        assert normalize_value(0, -10, 10) == 0.5
        assert normalize_value(-10, -10, 10) == 0.0
        assert normalize_value(10, -10, 10) == 1.0


class TestLogarithmicScale:
    """Tests for the logarithmic_scale function."""

    def test_logarithmic_scale_basic(self):
        """Test basic logarithmic scaling with standard inputs."""
        # Test with positive values
        assert logarithmic_scale(9) == math.log(10, 10)  # log_10(9+1)
        assert logarithmic_scale(99) == math.log(100, 10)  # log_10(99+1)
        assert logarithmic_scale(0) == 0  # log_10(0+1) = 0

    def test_logarithmic_scale_negative(self):
        """Test logarithmic scaling with negative values."""
        # Negative values should return 0
        assert logarithmic_scale(-1) == 0
        assert logarithmic_scale(-100) == 0

    def test_logarithmic_scale_custom_base(self):
        """Test logarithmic scaling with a custom base."""
        # Test with base 2
        assert logarithmic_scale(3, base=2) == math.log(4, 2)  # log_2(3+1)

        # Test with base e
        assert logarithmic_scale(1, base=math.e) == math.log(2, math.e)  # log_e(1+1)

    def test_logarithmic_scale_large_values(self):
        """Test logarithmic scaling with large values."""
        # Test with large values to ensure proper compression
        assert logarithmic_scale(999) == math.log(1000, 10)
        assert logarithmic_scale(9999) == math.log(10000, 10)

        # Verify large difference in input produces smaller difference in output
        large_diff_input = abs(logarithmic_scale(1000) - logarithmic_scale(100))
        small_diff_input = abs(logarithmic_scale(100) - logarithmic_scale(10))

        # In log scale, these differences should be approximately equal but not exactly equal
        # due to the nature of logarithmic scaling.
        assert abs(large_diff_input - small_diff_input) < 0.05


class TestScoreDocumentsWithSocialMetrics:
    """Tests for the score_documents_with_social_metrics function."""

    @pytest.fixture
    def sample_docs(self):
        """Create a set of sample documents with metadata."""
        return [
            Document(
                page_content="Doc 1",
                metadata={
                    "timestamp_unix": 1620000000.0,
                    "viewCount": 1000,
                    "likeCount": 100,
                    "retweetCount": 50,
                    "url": "https://example.com/1",
                },
            ),
            Document(
                page_content="Doc 2",
                metadata={
                    "timestamp_unix": 1620001000.0,
                    "viewCount": 500,
                    "likeCount": 200,
                    "retweetCount": 30,
                    "url": "https://example.com/2",
                },
            ),
            Document(
                page_content="Doc 3",
                metadata={
                    "timestamp_unix": 1620002000.0,
                    "viewCount": 2000,
                    "likeCount": 50,
                    "retweetCount": 100,
                    "url": "https://example.com/3",
                },
            ),
        ]

    def test_score_documents_basic(self, sample_docs):
        """Test basic document scoring with default weights."""
        # Define weights
        weights = {
            "recency_weight": 0.4,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.2,
        }

        # Score documents
        scored_docs = score_documents_with_social_metrics(sample_docs, weights)

        # Basic assertions
        assert len(scored_docs) == 3

        # Documents should be sorted by descending score
        scores = [score for _, score in scored_docs]
        assert scores[0] >= scores[1] >= scores[2]

        # The most recent document with high view count should be scored highest
        assert scored_docs[0][0].page_content == "Doc 3"

        # Verify format of result
        for doc, score in scored_docs:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_score_documents_custom_weights(self, sample_docs):
        """Test document scoring with custom weights."""
        # Prioritize recency heavily
        recency_weights = {
            "recency_weight": 0.9,
            "view_weight": 0.05,
            "like_weight": 0.025,
            "retweet_weight": 0.025,
        }

        # Score with recency emphasis
        recency_scored_docs = score_documents_with_social_metrics(sample_docs, recency_weights)

        # Most recent doc should be first
        assert recency_scored_docs[0][0].page_content == "Doc 3"

        # Prioritize likes heavily
        like_weights = {
            "recency_weight": 0.05,
            "view_weight": 0.05,
            "like_weight": 0.85,
            "retweet_weight": 0.05,
        }

        # Score with likes emphasis
        like_scored_docs = score_documents_with_social_metrics(sample_docs, like_weights)

        # Doc with most likes should be first
        assert like_scored_docs[0][0].page_content == "Doc 2"

    def test_score_documents_missing_metadata(self):
        """Test document scoring with missing metadata."""
        # Create docs with missing fields
        docs_with_missing = [
            Document(
                page_content="Complete",
                metadata={
                    "timestamp_unix": 1620000000.0,
                    "viewCount": 1000,
                    "likeCount": 100,
                    "retweetCount": 50,
                },
            ),
            Document(
                page_content="Missing views",
                metadata={
                    "timestamp_unix": 1620001000.0,
                    # Missing viewCount
                    "likeCount": 200,
                    "retweetCount": 30,
                },
            ),
            Document(
                page_content="Only timestamp",
                metadata={
                    "timestamp_unix": 1620002000.0,
                    # Missing all engagement metrics
                },
            ),
        ]

        weights = {
            "recency_weight": 0.4,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.2,
        }

        # Score documents
        scored_docs = score_documents_with_social_metrics(docs_with_missing, weights)

        # Should still work with missing metadata
        assert len(scored_docs) == 3

        # Check that default values were used for missing fields
        for doc, _ in scored_docs:
            if doc.page_content == "Missing views":
                assert "_scoring" in doc.metadata
                assert doc.metadata["_scoring"]["view_score"] == 0  # Default to 0 when missing

            if doc.page_content == "Only timestamp":
                assert "_scoring" in doc.metadata
                # All engagement scores should be 0
                assert doc.metadata["_scoring"]["view_score"] == 0
                assert doc.metadata["_scoring"]["like_score"] == 0
                assert doc.metadata["_scoring"]["retweet_score"] == 0

    def test_score_documents_empty_list(self):
        """Test document scoring with an empty document list."""
        weights = {
            "recency_weight": 0.4,
            "view_weight": 0.2,
            "like_weight": 0.2,
            "retweet_weight": 0.2,
        }

        # Score empty list
        scored_docs = score_documents_with_social_metrics([], weights)

        # Should return empty list
        assert scored_docs == []

    def test_score_documents_logarithmic_scaling(self, sample_docs):
        """Test that the function applies logarithmic scaling to engagement metrics."""
        with patch("app.rag.scoring.logarithmic_scale", wraps=logarithmic_scale) as mock_log_scale:
            weights = {
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            }

            # Score documents
            score_documents_with_social_metrics(sample_docs, weights)

            # Verify logarithmic_scale was called for each engagement metric
            assert mock_log_scale.call_count >= 9  # 3 docs * 3 metrics


class TestDiversifyRankedDocuments:
    """Tests for the diversify_ranked_documents function."""

    @pytest.fixture
    def sample_scored_docs(self):
        """Create a set of sample scored documents."""
        return [
            (Document(page_content="Doc 1", metadata={"url": "https://example.com/1"}), 0.9),
            (Document(page_content="Doc 2", metadata={"url": "https://example.com/2"}), 0.8),
            (Document(page_content="Doc 3", metadata={"url": "https://example.com/3"}), 0.7),
            (Document(page_content="Doc 4", metadata={"url": "https://example.com/4"}), 0.6),
            (Document(page_content="Doc 5", metadata={"url": "https://example.com/5"}), 0.5),
            (Document(page_content="Doc 6", metadata={"url": "https://example.com/6"}), 0.4),
            (Document(page_content="Doc 7", metadata={"url": "https://example.com/7"}), 0.3),
            (Document(page_content="Doc 8", metadata={"url": "https://example.com/8"}), 0.2),
            (Document(page_content="Doc 9", metadata={"url": "https://example.com/9"}), 0.1),
            (Document(page_content="Doc 10", metadata={"url": "https://example.com/10"}), 0.05),
        ]

    def test_diversify_no_diversity(self, sample_scored_docs):
        """Test with diversity_factor=0 (pure ranking)."""
        # With diversity_factor=0, should just take top k
        k = 5
        result = diversify_ranked_documents(sample_scored_docs, k, diversity_factor=0)

        # Should get exactly k results
        assert len(result) == k

        # Should be top k documents in order
        expected_docs = [doc for doc, _ in sample_scored_docs[:k]]
        assert result == expected_docs

    def test_diversify_max_diversity(self, sample_scored_docs):
        """Test with diversity_factor=1 (maximum diversity)."""
        # With diversity_factor=1, should take 0 documents directly and
        # rest with diversity algorithm
        k = 5
        result = diversify_ranked_documents(sample_scored_docs, k, diversity_factor=1.0)

        # Should get exactly k results
        assert len(result) == k

        # First document should be highest ranked (even with max diversity)
        assert result[0].page_content == "Doc 1"

        # Should include some documents from lower ranks due to diversity
        content_set = set(doc.page_content for doc in result)
        assert any(f"Doc {i}" in content_set for i in range(5, 11))

    def test_diversify_mixed_strategy(self, sample_scored_docs):
        """Test with a mixed diversity strategy."""
        # With diversity_factor=0.6, should take 40% directly, 60% with diversity
        k = 5
        diversity_factor = 0.6
        result = diversify_ranked_documents(
            sample_scored_docs, k, diversity_factor=diversity_factor
        )

        # Should get exactly k results
        assert len(result) == k

        # Should include top documents based on (1-diversity_factor)*k
        direct_count = int((1 - diversity_factor) * k)
        top_docs = [doc.page_content for doc, _ in sample_scored_docs[:direct_count]]

        for doc in top_docs:
            assert doc in [d.page_content for d in result]

    def test_diversify_k_larger_than_docs(self, sample_scored_docs):
        """Test when k is larger than the number of documents."""
        # Request more documents than are available
        k = 15
        result = diversify_ranked_documents(sample_scored_docs, k)

        # Should return all available documents
        assert len(result) == 10

        # Should contain all documents
        doc_contents = [doc.page_content for doc in result]
        for i in range(1, 11):
            assert f"Doc {i}" in doc_contents

    def test_diversify_empty_list(self):
        """Test with an empty document list."""
        # Empty input should return empty output
        result = diversify_ranked_documents([], k=5)
        assert result == []

    def test_diversify_k_zero(self, sample_scored_docs):
        """Test with k=0."""
        # k=0 should return empty list
        result = diversify_ranked_documents(sample_scored_docs, k=0)
        assert result == []
