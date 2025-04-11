"""
Tests for request logging middleware.

This module contains tests for the middleware components in the app,
particularly the log_requests middleware that handles request logging,
timing information, and request ID tracking.
"""

import logging
import time
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import Request, Response

from app.middleware.request_logging import log_requests


class TestLogRequests:
    """Tests for the log_requests middleware function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create a mock Request object."""
        mock_req = MagicMock(spec=Request)
        mock_req.method = "GET"
        mock_req.url.path = "/api/v1/status"
        mock_req.client.host = "127.0.0.1"
        mock_req.headers = {}
        return mock_req

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create a mock Response object."""
        mock_resp = MagicMock(spec=Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        return mock_resp

    @pytest.fixture
    def mock_call_next(self, mock_response) -> AsyncMock:
        """Create a mock for the call_next function."""
        async_mock = AsyncMock()
        async_mock.return_value = mock_response
        return async_mock

    @pytest.mark.asyncio
    async def test_log_requests_happy_path(self, mock_request, mock_call_next, caplog):
        """Test normal, successful request logging."""
        # Set up logger capture
        caplog.set_level(logging.INFO)

        # Set a fixed UUID for testing
        test_uuid = "test-uuid-1234"

        # Run the middleware with patches
        with patch(
            "app.middleware.request_logging.uuid4",
            return_value=uuid.UUID(int=0x12345678123456781234567812345678),
        ):
            response = await log_requests(mock_request, mock_call_next)

            # Verify the call_next was called with the request
            mock_call_next.assert_called_once_with(mock_request)

            # Verify response headers were added
            assert "X-Request-ID" in response.headers
            assert "X-Processing-Time" in response.headers
            assert float(response.headers["X-Processing-Time"]) >= 0

            # Verify log messages were created
            assert "START Request: GET /api/v1/status" in caplog.text
            assert "END Request: Status=200" in caplog.text

    @pytest.mark.asyncio
    async def test_log_requests_with_existing_request_id(
        self, mock_request, mock_call_next, caplog
    ):
        """Test request logging with an existing X-Request-ID header."""
        # Set up logger capture
        caplog.set_level(logging.INFO)

        # Set a specific request ID in the headers
        existing_request_id = "existing-id-5678"
        mock_request.headers = {"X-Request-ID": existing_request_id}

        # Run the middleware
        response = await log_requests(mock_request, mock_call_next)

        # Verify the existing request ID was preserved
        assert response.headers["X-Request-ID"] == existing_request_id

        # Verify log messages used the existing ID
        assert f"rid={existing_request_id}" in caplog.text

    @pytest.mark.asyncio
    async def test_log_requests_exception_handling(self, mock_request, caplog):
        """Test request logging when the call_next function raises an exception."""
        # Set up logger capture
        caplog.set_level(logging.INFO)

        # Create a call_next mock that raises an exception
        error_call_next = AsyncMock(side_effect=ValueError("Test error"))

        # Run the middleware and expect an exception
        with pytest.raises(ValueError) as exc_info:
            await log_requests(mock_request, error_call_next)

        # Verify the exception was re-raised
        assert str(exc_info.value) == "Test error"

        # Verify log messages were created, including error
        assert "START Request: GET /api/v1/status" in caplog.text
        assert "FAIL Request: Unhandled exception" in caplog.text

    @pytest.mark.asyncio
    async def test_log_requests_timing_accuracy(self, mock_request, mock_call_next):
        """Test that the timing information is accurate."""

        # Create a slow call_next that introduces a delay
        async def delayed_call_next(request):
            await asyncio.sleep(0.1)  # 100ms delay
            return mock_call_next.return_value

        # Run the middleware
        with patch("app.middleware.request_logging.time.time") as mock_time:
            # Mock time.time() to return specific values on consecutive calls
            mock_time.side_effect = [1000.0, 1000.1]  # 100ms difference

            response = await log_requests(mock_request, mock_call_next)

            # Verify the processing time header is accurate
            assert float(response.headers["X-Processing-Time"]) == 0.1

    @pytest.mark.asyncio
    async def test_log_requests_no_client(self, mock_request, mock_call_next, caplog):
        """Test request logging when the client information is missing."""
        # Set up logger capture
        caplog.set_level(logging.INFO)

        # Set client to None to simulate missing client info
        mock_request.client = None

        # Run the middleware
        response = await log_requests(mock_request, mock_call_next)

        # Verify log message uses 'unknown' for client
        assert "Client=unknown" in caplog.text


# Need to import asyncio for the timing test
import asyncio
