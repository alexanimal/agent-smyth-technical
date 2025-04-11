# Prompts Module

This module manages the prompt templates used throughout the RAG system, providing centralized management and access to prompt templates optimized for different query types.

## Overview

The prompts module is responsible for:
- Defining and managing prompt templates
- Dynamically constructing prompts based on query type and context
- Implementing advanced prompting techniques
- Optimizing prompt parameters for different LLM models

## Components

### PromptManager

The core class that provides access to all prompt templates:

```python
from app.prompts import PromptManager

# Access classification prompt
classification_prompt = PromptManager.get_classification_prompt()

# Access generation prompt
generation_prompt = PromptManager.get_generation_prompt(
    query_type="technical",
    include_alternative_viewpoint=True
)
```

### Prompt Types

The module includes specialized prompts for different stages of the RAG pipeline:

- **Classification Prompts**: For determining query type (technical, investment, etc.)
- **Retrieval Prompts**: For optimizing document retrieval
- **Generation Prompts**: For generating responses with source attribution
- **Alternative Viewpoint Prompts**: For creating balanced perspectives

## Advanced Prompting Techniques

The module implements several advanced prompting techniques:

### Chain-of-Thought (CoT)

Guides the model through a step-by-step reasoning process, particularly useful for complex financial analysis:

```python
# Example CoT template structure
cot_template = """
To answer this question about {topic}, I'll reason through it step by step:
1. First, I'll analyze the key metrics mentioned in the context
2. Then, I'll evaluate recent market trends
3. Finally, I'll synthesize this information into a comprehensive answer

{context}

Question: {query}

Step 1: Analyzing key metrics...
"""
```

### Self-Evaluation (SE)

Prompts the model to assess its own confidence, improving accuracy:

```python
# Example self-evaluation template
se_template = """
{context}

Question: {query}

Let me generate an answer:
{answer}

Now I'll evaluate the quality of my answer:
- Completeness: Did I address all aspects of the question?
- Accuracy: Is my answer supported by the provided context?
- Confidence: How certain am I about this answer?
"""
```

### Tree-of-Thought (ToT)

Enables the model to explore multiple reasoning paths for complex financial queries:

```python
# Example ToT structure
tot_template = """
Given this question about investment strategy, I'll explore multiple analysis paths:

Path A: Technical Analysis
{technical_analysis_reasoning}

Path B: Fundamental Analysis
{fundamental_analysis_reasoning}

Path C: Market Sentiment Analysis
{sentiment_analysis_reasoning}

Based on these analyses, my comprehensive answer is:
"""
```

## Prompt Optimization

The module includes functionality for:
- Adjusting prompt parameters based on model (GPT-3.5, GPT-4, etc.)
- Balancing prompt length vs. detail for optimal token usage
- Including relevant context based on query type

## Usage Examples

### Classification

```python
from app.prompts import PromptManager

classification_prompt = PromptManager.get_classification_prompt()
chain = classification_prompt | model | output_parser
result = await chain.ainvoke({"query": "What are investors saying about AAPL?"})
```

### Response Generation

```python
from app.prompts import PromptManager

prompt = PromptManager.get_generation_prompt(
    query_type="investment",
    include_alternative_viewpoint=True
)
chain = prompt | model | output_parser
result = await chain.ainvoke({
    "query": "What are investors saying about AAPL?",
    "context": documents,
    "sources": source_urls
})
```

## Extending the Module

To add new prompt templates:

1. Define the template in the appropriate section
2. Add accessor method to the `PromptManager` class
3. Include any specialized formatting or processing logic needed

Example:

```python
@classmethod
def get_custom_prompt(cls, parameter1, parameter2):
    """Get a custom prompt template with the specified parameters."""
    template = """
    This is a custom prompt with {parameter1} and {parameter2}.

    {context}

    Question: {query}
    """

    return ChatPromptTemplate.from_template(template)
```

## Best Practices

- Keep prompts declarative and focused
- Include clear instructions at the beginning
- Provide examples for complex tasks (few-shot learning)
- Include explicit output format requirements
- Balance detail vs. token efficiency
