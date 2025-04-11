# Prompts Package

This package contains prompt templates and generation logic for the various LLM interactions in the RAG Agent application.

## Contents

- `analysis.py`: Defines templates for investment and technical analysis prompts.
- `classification.py`: Contains templates for query classification.
- `generator.py`: Implements the generic `PromptGenerator` class for constructing prompts with dynamic insertion of context.
- `__init__.py`: Re-exports prompt components for backward compatibility.

## Usage

```python
from app.prompts import PromptGenerator, get_investment_analysis_prompt

# Create a custom prompt with dynamic context
prompt_generator = PromptGenerator("Base template with {placeholder}")
formatted_prompt = prompt_generator.generate(placeholder="inserted value")

# Or use pre-defined prompt templates
investment_prompt = get_investment_analysis_prompt(query="analysis request", context="relevant documents")
```

## Key Features

- Template-based prompt generation with parameter substitution
- Specialized prompt templates for different query types
- Consistent prompt structure across the application
- Support for RAG-specific prompt patterns (query + context)
