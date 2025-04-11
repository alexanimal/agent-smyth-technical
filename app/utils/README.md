# Utils Package

This package contains utility functions and helper modules that are used throughout the RAG Agent application.

## Contents

- `logging.py`: Provides logging configuration and utility functions.
- `document.py`: Contains utilities for document processing, extraction, and manipulation.
- `sanitization.py`: Implements functions for sanitizing user inputs and processing outputs.
- `errors.py`: Defines custom exception classes and error handling utilities.
- `__init__.py`: Re-exports utility functions for convenient imports.

## Usage

```python
from app.utils import setup_logging, sanitize_input, extract_sources
from app.utils.errors import QueryProcessingError

# Set up logging for a module
logger = setup_logging(__name__)

# Process user input safely
cleaned_input = sanitize_input(user_query)

# Handle errors consistently
try:
    # Your code here
    pass
except Exception as e:
    raise QueryProcessingError("Failed to process query") from e
```

## Key Features

- Centralized logging configuration
- Input sanitization and validation
- Document processing utilities
- Error handling patterns
- Common helper functions used across the application
