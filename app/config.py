"""
Configuration module for the RAG Agent application.

This module handles loading and validating environment variables, setting up logging,
and initializing Sentry for error tracking. It provides a centralized configuration
management system through the Settings class.

Typical usage:
    from app.config import settings

    # Access configuration values
    model = settings.model_name
    api_key = settings.openai_api_key
"""

import logging
import os
from typing import List

import sentry_sdk
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class uses Pydantic's BaseSettings to load and validate
    environment variables, with fallbacks to a .env file.
    It handles configuration for API keys, model selection,
    logging, and external services.

    Attributes:
        openai_api_key (str): OpenAI API key for LLM access
        api_key (str, optional): API key for authenticating requests
        environment (str): Deployment environment (development, staging, production)
        sentry_dsn (str, optional): Sentry DSN for error tracking
        model_name (str): Name of the OpenAI model to use
        log_level (str): Logging verbosity level
        mocks_dir_name (str): Directory name for mock data storage
        cors_origins (List[str]): List of allowed origins for CORS
    """

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    api_key: str | None = Field(None, alias="API_KEY")  # For production auth
    environment: str = Field("development", alias="ENVIRONMENT")
    sentry_dsn: str | None = Field(None, alias="SENTRY_DSN")
    model_name: str = Field("gpt-4o-mini", alias="MODEL_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    mocks_dir_name: str = Field("data", alias="MOCKS_DIR_NAME")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"], alias="CORS_ORIGINS"
    )

    class Config:
        """
        Configuration for the Settings class.

        Specifies environment file location, encoding, and how to handle
        extra environment variables.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


def setup_logging_and_sentry(settings: Settings):
    """
    Configure logging and initialize Sentry if DSN is provided.

    Sets up the Python logging system with the configured log level and
    initializes Sentry for error tracking if in production environment
    and a Sentry DSN is provided.

    Args:
        settings (Settings): Application settings including log level and Sentry configuration

    Returns:
        None
    """
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level.upper()}")

    if settings.sentry_dsn and settings.environment == "production":
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            send_default_pii=True,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            environment=settings.environment,
            # Consider adding integrations like FastAPIIntegration
        )
        logger.info("Sentry initialized.")
    elif settings.sentry_dsn:
        logger.warning(
            "Sentry DSN found but environment is not 'production'. Sentry not initialized."
        )
    else:
        logger.info("Sentry DSN not found. Sentry not initialized.")


# Load .env file
load_dotenv()

# Create settings instance
settings = Settings()  # type: ignore[call-arg]

# Perform initial checks
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file!")

# Define project root based on this file's location
# Assumes config.py is in src directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCKS_DIR_PATH = os.path.join(PROJECT_ROOT, settings.mocks_dir_name)

logger = logging.getLogger(__name__)  # Re-get logger after basicConfig
logger.info(f"Project Root determined as: {PROJECT_ROOT}")
logger.info(f"Mocks Directory determined as: {MOCKS_DIR_PATH}")
