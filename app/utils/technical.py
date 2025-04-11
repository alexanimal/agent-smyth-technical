"""
Technical analysis utilities for financial queries.

This module provides utilities for extracting and analyzing technical indicators
and chart patterns from documents to enhance responses to technical analysis queries.
"""

import re
from typing import Dict, List, Set

from langchain_core.documents import Document


async def get_technical_indicators(message: str, docs: List[Document]) -> str:
    """
    Extract and analyze technical indicators from the query and documents.

    This method identifies potential ticker symbols in the query, extracts
    technical indicator information from the documents, and formats it
    for inclusion in the context.

    Args:
        message: The user's query message
        docs: List of retrieved documents

    Returns:
        str: Formatted technical indicator data
    """
    # Extract potential ticker symbols from the message
    # This is a simplified approach - in production, you would use a more robust method
    potential_tickers = set(re.findall(r"\$?[A-Z]{1,5}", message))

    # Extract technical indicators from documents
    technical_data: Dict[str, List[str]] = {}
    patterns: Dict[str, List[str]] = {}
    for doc in docs:
        content = doc.page_content.lower()

        # Extract moving averages
        ma_matches = re.findall(r"(sma|ema)[\s-]*(\d+)[^\d]", content)
        for ma_type, period in ma_matches:
            key = f"{ma_type.upper()}{period}"
            if key not in technical_data:
                technical_data[key] = []
            technical_data[key].append(doc.metadata.get("url", "unknown"))

        # Extract other indicators
        indicators = {
            "rsi": r"rsi[\s-]*(\d+)[^\d]",
            "macd": r"macd",
            "stochastic": r"stoch(astic)?",
            "bollinger": r"bollinger",
            "volume": r"volume",
        }

        for ind_name, pattern in indicators.items():
            if re.search(pattern, content):
                if ind_name not in technical_data:
                    technical_data[ind_name] = []
                technical_data[ind_name].append(doc.metadata.get("url", "unknown"))

        # Extract chart patterns
        chart_patterns = [
            "head and shoulders",
            "double top",
            "double bottom",
            "triangle",
            "wedge",
            "flag",
            "pennant",
            "cup and handle",
        ]

        for pattern in chart_patterns:
            if pattern in content:
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(doc.metadata.get("url", "unknown"))

    # Format the technical data
    result = "Technical Indicator Data:\n"

    if technical_data:
        result += "Indicators mentioned in context:\n"
        for indicator, sources in technical_data.items():
            result += f"- {indicator}: mentioned in {len(sources)} sources\n"
    else:
        result += "No specific technical indicators found in the context.\n"

    if patterns:
        result += "\nChart Patterns mentioned in context:\n"
        for pattern, sources in patterns.items():
            result += f"- {pattern.title()}: mentioned in {len(sources)} sources\n"
    else:
        result += "\nNo specific chart patterns found in the context.\n"

    if potential_tickers:
        result += f"\nPotential tickers identified: {', '.join(potential_tickers)}\n"

    return result
