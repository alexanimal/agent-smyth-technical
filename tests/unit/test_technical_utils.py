"""
Tests for technical analysis utilities.

This module contains tests for the technical analysis utility functions
in app/utils/technical.py, focusing on the extraction and analysis of
technical indicators from documents.
"""

from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from app.utils.technical import get_technical_indicators


class TestGetTechnicalIndicators:
    """Tests for the get_technical_indicators function."""

    @pytest.mark.asyncio
    async def test_extract_moving_averages(self):
        """Test extracting moving average indicators from documents."""
        # Create a message asking about moving averages
        message = "What do the SMA-50 and EMA-20 indicate for AAPL?"

        # Create test documents with moving average references
        docs = [
            Document(
                page_content="AAPL's SMA-50 is at $190, showing an uptrend.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="The EMA-20 for AAPL crossed above the SMA-50, which is a bullish signal.",
                metadata={"url": "source2"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "Indicators mentioned in context" in result
        assert "SMA50" in result
        assert "EMA20" in result

    @pytest.mark.asyncio
    async def test_extract_other_indicators(self):
        """Test extracting other technical indicators (RSI, MACD, etc.)."""
        # Create a message asking about RSI and MACD
        message = "What's the RSI and MACD for AAPL showing?"

        # Create test documents with indicator references
        docs = [
            Document(
                page_content="AAPL's RSI is currently at 70, approaching overbought territory.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="The MACD for AAPL shows a bullish crossover, indicating positive momentum.",
                metadata={"url": "source2"},
            ),
            Document(
                page_content="Stochastic oscillator for AAPL is also showing overbought conditions.",
                metadata={"url": "source3"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "Indicators mentioned in context" in result
        assert "rsi" in result.lower()
        assert "macd" in result.lower()
        assert "stochastic" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_chart_patterns(self):
        """Test extracting chart patterns from documents."""
        # Create a message asking about chart patterns
        message = "Are there any chart patterns visible for AAPL?"

        # Create test documents with chart pattern references
        docs = [
            Document(
                page_content="AAPL is forming a head and shoulders pattern on the daily chart.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="There's a potential double bottom forming on AAPL's hourly chart.",
                metadata={"url": "source2"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "Chart Patterns mentioned in context" in result
        assert "Head And Shoulders" in result
        assert "Double Bottom" in result

    @pytest.mark.asyncio
    async def test_extract_ticker_symbols(self):
        """Test extracting ticker symbols from the message."""
        # Create a message with ticker symbols
        message = "Compare the RSI for $AAPL and MSFT please"

        # Create simple test documents
        docs = [
            Document(page_content="AAPL's RSI is currently at 65.", metadata={"url": "source1"}),
            Document(page_content="MSFT's RSI is at 55.", metadata={"url": "source2"}),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "Potential tickers identified" in result
        assert "AAPL" in result
        assert "MSFT" in result

    @pytest.mark.asyncio
    async def test_no_indicators_found(self):
        """Test behavior when no indicators are found in documents."""
        # Create a general message
        message = "What's the technical outlook for AAPL?"

        # Create test documents with no specific indicators
        docs = [
            Document(
                page_content="AAPL stock has been performing well recently.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="Analysts are positive about AAPL's future.",
                metadata={"url": "source2"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "No specific technical indicators found" in result
        assert "No specific chart patterns found" in result
        assert "AAPL" in result  # Should still extract ticker

    @pytest.mark.asyncio
    async def test_multiple_indicators_same_type(self):
        """Test extracting multiple indicators of the same type."""
        # Create a message asking about moving averages
        message = "What do the various moving averages show for AAPL?"

        # Create test documents with multiple MA references
        docs = [
            Document(
                page_content="AAPL's SMA-50, SMA-100, and SMA-200 are all trending upward.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="The EMA-9 and EMA-20 are both above the SMA-50.",
                metadata={"url": "source2"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result
        assert "SMA50" in result
        assert "SMA100" in result
        assert "SMA200" in result
        assert "EMA9" in result
        assert "EMA20" in result

    @pytest.mark.asyncio
    async def test_empty_docs(self):
        """Test behavior with empty document list."""
        # Create a simple message
        message = "What's the RSI for AAPL?"

        # Get technical indicators with empty docs
        result = await get_technical_indicators(message, [])

        # Assertions
        assert "Technical Indicator Data" in result
        assert "No specific technical indicators found" in result
        assert "AAPL" in result  # Should still extract ticker

    @pytest.mark.asyncio
    async def test_multiple_indicators_multiple_documents(self):
        """Test extracting various indicators across multiple documents."""
        # Create a comprehensive message
        message = "Give me a complete technical analysis for $TSLA including moving averages, RSI, and MACD"

        # Create test documents with various indicators
        docs = [
            Document(
                page_content="TSLA's SMA-50 is at $250, while the SMA-200 is at $230.",
                metadata={"url": "source1"},
            ),
            Document(
                page_content="The RSI for TSLA is at 58, in neutral territory.",
                metadata={"url": "source2"},
            ),
            Document(
                page_content="TSLA's MACD is showing a bullish crossover with increasing momentum.",
                metadata={"url": "source3"},
            ),
            Document(
                page_content="Volume for TSLA has been above average in recent trading sessions.",
                metadata={"url": "source4"},
            ),
            Document(
                page_content="A potential cup and handle pattern is forming on TSLA's daily chart.",
                metadata={"url": "source5"},
            ),
        ]

        # Get technical indicators
        result = await get_technical_indicators(message, docs)

        # Assertions
        assert "Technical Indicator Data" in result

        # Should contain indicators
        assert "SMA50" in result
        assert "SMA200" in result
        assert "rsi" in result.lower()
        assert "macd" in result.lower()
        assert "volume" in result.lower()

        # Should contain chart patterns
        assert "Chart Patterns" in result
        assert "Cup And Handle" in result

        # Should identify ticker
        assert "TSLA" in result
