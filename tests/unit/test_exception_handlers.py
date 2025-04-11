"""
Unit tests for FastAPI exception handlers.

This module tests the custom exception handlers used in the application
to ensure proper formatting of error responses.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY

from app.config import settings
from app.middleware.exception_handlers import (
    ModelValidationError,
    add_exception_handlers,
    model_validation_exception_handler,
    validation_exception_handler,
)


class TestModelValidationExceptionHandler:
    """Tests for the model_validation_exception_handler function."""

    @pytest.mark.asyncio
    async def test_model_validation_error_response(self):
        """Test that ModelValidationError returns a properly formatted response."""
        # Create a mock request
        mock_request = Mock(spec=Request)

        # Create a ModelValidationError with a message
        error_message = "Invalid model: xyz-invalid-model"
        error = ModelValidationError(error_message)

        # Patch the settings with allowed models
        with patch.object(settings, "allowed_models", ["gpt-4", "gpt-3.5-turbo"]):
            with patch.object(settings, "default_model", "gpt-4"):
                # Call the exception handler
                response = await model_validation_exception_handler(mock_request, error)

                # Verify the response status code
                assert response.status_code == HTTP_400_BAD_REQUEST

                # Verify response JSON content
                response_body = json.loads(response.body)
                assert response_body["detail"] == error_message
                assert response_body["type"] == "model_validation_error"
                assert "allowed_models" in response_body
                assert "gpt-4" in response_body["allowed_models"]
                assert "gpt-3.5-turbo" in response_body["allowed_models"]
                assert response_body["default_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_model_validation_logging(self):
        """Test that ModelValidationError properly logs the error."""
        # Create a mock request
        mock_request = Mock(spec=Request)

        # Create a ModelValidationError with a message
        error_message = "Invalid model: xyz-invalid-model"
        error = ModelValidationError(error_message)

        # Patch the logger
        with patch("app.middleware.exception_handlers.logger") as mock_logger:
            # Call the exception handler
            response = await model_validation_exception_handler(mock_request, error)

            # Verify the warning was logged
            mock_logger.warning.assert_called_once()
            # Check the log message contains the error message
            assert error_message in mock_logger.warning.call_args[0][0]


class TestValidationExceptionHandler:
    """Tests for the validation_exception_handler function."""

    @pytest.mark.asyncio
    async def test_validation_error_response_with_body(self):
        """Test that RequestValidationError returns a properly formatted response with body."""
        # Create a mock request
        mock_request = Mock(spec=Request)

        # Create a RequestValidationError with validation errors and body
        errors = [
            {"loc": ["body", "message"], "msg": "Field required", "type": "value_error.missing"}
        ]
        body = {"incomplete": "data"}

        # Create a mock RequestValidationError
        mock_error = Mock(spec=RequestValidationError)
        mock_error.errors.return_value = errors
        mock_error.body = body

        # Call the exception handler
        with patch("app.middleware.exception_handlers.logger") as mock_logger:
            response = await validation_exception_handler(mock_request, mock_error)

            # Verify the response status code
            assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

            # Verify response JSON content
            response_body = json.loads(response.body)
            assert response_body["detail"] == errors
            assert response_body["type"] == "validation_error"
            assert response_body["body"] == body

            # Verify logging
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_error_response_without_body(self):
        """Test that RequestValidationError returns a properly formatted response without body."""
        # Create a mock request
        mock_request = Mock(spec=Request)

        # Create a RequestValidationError with validation errors but no body
        errors = [
            {
                "loc": ["query", "model"],
                "msg": "String too short",
                "type": "value_error.any_str.min_length",
                "ctx": {"limit_value": 1},
            }
        ]

        # Create a mock RequestValidationError without a body attribute
        mock_error = Mock(spec=RequestValidationError)
        mock_error.errors.return_value = errors
        # No body attribute

        # Call the exception handler
        with patch("app.middleware.exception_handlers.logger") as mock_logger:
            response = await validation_exception_handler(mock_request, mock_error)

            # Verify the response status code
            assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

            # Verify response JSON content
            response_body = json.loads(response.body)
            assert response_body["detail"] == errors
            assert response_body["type"] == "validation_error"
            assert response_body["body"] is None

            # Verify logging
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_complex_validation_error(self):
        """Test handling of complex validation errors with nested fields."""
        # Create a mock request
        mock_request = Mock(spec=Request)

        # Create a complex set of validation errors
        errors = [
            {
                "loc": ["body", "query", "advanced_options", "temperature"],
                "msg": "Value must be between 0 and 1",
                "type": "value_error.number.not_gt",
                "ctx": {"limit_value": 0},
            },
            {
                "loc": ["body", "model"],
                "msg": "Model not in allowed list",
                "type": "value_error.str.not_in",
                "ctx": {"allowed_values": ["gpt-4", "gpt-3.5-turbo"]},
            },
        ]

        # Create a sample request body
        body = {
            "query": {
                "text": "What is the price of ABC stock?",
                "advanced_options": {"temperature": -0.5},
            },
            "model": "invalid-model",
        }

        # Create a mock RequestValidationError
        mock_error = Mock(spec=RequestValidationError)
        mock_error.errors.return_value = errors
        mock_error.body = body

        # Call the exception handler
        response = await validation_exception_handler(mock_request, mock_error)

        # Verify the response
        response_body = json.loads(response.body)
        assert response_body["detail"] == errors
        assert response_body["body"] == body


class TestAddExceptionHandlers:
    """Tests for the add_exception_handlers function."""

    def test_add_exception_handlers(self):
        """Test that exception handlers are correctly added to the FastAPI app."""
        # Create a mock FastAPI app
        mock_app = Mock(spec=FastAPI)

        # Call the function
        add_exception_handlers(mock_app)

        # Verify the app's add_exception_handler method was called for each handler
        assert mock_app.add_exception_handler.call_count == 2

        # Verify the handlers were registered with the correct exception types
        calls = [
            call(ModelValidationError, model_validation_exception_handler),
            call(RequestValidationError, validation_exception_handler),
        ]
        mock_app.add_exception_handler.assert_has_calls(calls, any_order=True)

    def test_add_exception_handlers_with_real_app(self):
        """Test with a real FastAPI app to ensure no errors occur during registration."""
        # Create a real FastAPI app
        app = FastAPI()

        # Mock the add_exception_handler method to verify it's called correctly
        original_add_handler = app.add_exception_handler
        calls = []

        def mock_add_exception_handler(exc_class_or_status_code, handler):
            calls.append((exc_class_or_status_code, handler))
            # Call the original method to maintain functionality
            return original_add_handler(exc_class_or_status_code, handler)

        app.add_exception_handler = mock_add_exception_handler

        # This should not raise any exceptions
        add_exception_handlers(app)

        # Verify the handlers were registered correctly
        assert len(calls) == 2

        # Check that ModelValidationError was registered with model_validation_exception_handler
        model_validation_call = next(
            (call for call in calls if call[0] == ModelValidationError), None
        )
        assert model_validation_call is not None
        assert model_validation_call[1] == model_validation_exception_handler

        # Check that RequestValidationError was registered with validation_exception_handler
        request_validation_call = next(
            (call for call in calls if call[0] == RequestValidationError), None
        )
        assert request_validation_call is not None
        assert request_validation_call[1] == validation_exception_handler
