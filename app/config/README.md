# Configuration Package

This package contains configuration management and environment variable handling for the RAG Agent application.

## Contents

- `settings.py`: Defines the `Settings` class for loading and validating environment variables, implementing configuration defaults, and setting up logging and monitoring.
- `__init__.py`: Re-exports configuration components for backward compatibility.

## Usage

```python
from app.config import settings

# Access configuration values
model_name = settings.model_name
api_key = settings.api_key
```

## Key Features

- Environment variable loading with validation using Pydantic
- Fallback to .env file when environment variables are not set
- Centralized configuration management
- Sentry integration for error tracking in production environments
