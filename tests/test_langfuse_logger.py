"""
Unit tests for Langfuse Logger (Component 4)

Uses mocking to avoid real Langfuse API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from utils.langfuse_logger import (
    is_langfuse_enabled,
    get_langfuse_client,
    log_followups,
    log_query_response,
    create_session_id,
    flush
)


class TestIsLangfuseEnabled:
    """Tests for is_langfuse_enabled() function"""

    def test_returns_false_when_no_keys(self):
        """Should return False when env vars not set"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            assert is_langfuse_enabled() is False

    def test_returns_false_when_partial_keys(self):
        """Should return False when only some keys set"""
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk"}, clear=True):
            assert is_langfuse_enabled() is False

    def test_returns_true_when_all_keys_set(self):
        """Should return True when all required keys set"""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test"
        }):
            assert is_langfuse_enabled() is True


class TestGetLangfuseClient:
    """Tests for get_langfuse_client() function"""

    def test_returns_none_when_disabled(self):
        """Should return None when Langfuse not configured"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            # Reset the global client
            import utils.langfuse_logger as logger
            logger._langfuse_client = None

            client = get_langfuse_client()
            assert client is None

    def test_creates_client_when_enabled(self):
        """Should create client when configured"""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test"
        }):
            # Reset the global client
            import utils.langfuse_logger as logger
            logger._langfuse_client = None

            with patch("langfuse.Langfuse") as mock_langfuse_class:
                mock_client = Mock()
                mock_langfuse_class.return_value = mock_client

                client = get_langfuse_client()

                mock_langfuse_class.assert_called_once()
                assert client == mock_client

    def test_reuses_existing_client(self):
        """Should reuse existing client instance"""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test"
        }):
            import utils.langfuse_logger as logger

            # Set a mock client directly
            mock_client = Mock()
            logger._langfuse_client = mock_client

            # Call get_langfuse_client
            client1 = get_langfuse_client()
            client2 = get_langfuse_client()

            # Should reuse existing client
            assert client1 == mock_client
            assert client2 == mock_client

            # Clean up
            logger._langfuse_client = None


class TestLogFollowups:
    """Tests for log_followups() function"""

    def test_returns_true_for_empty_list(self):
        """Should return True for empty followup list"""
        result = log_followups([], "query")
        assert result is True

    def test_returns_false_when_disabled(self):
        """Should return False when Langfuse not configured"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            import utils.langfuse_logger as logger
            logger._langfuse_client = None

            result = log_followups(["Q1?", "Q2?"], "query")
            assert result is False

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_creates_trace_with_followups(self, mock_get_client):
        """Should create trace and log follow-ups"""
        mock_client = Mock()
        mock_trace = Mock()
        mock_client.trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        result = log_followups(
            followups=["Question 1?", "Question 2?"],
            query="Original query",
            session_id="test-session"
        )

        assert result is True
        mock_client.trace.assert_called_once()
        # Should log each followup as generation
        assert mock_trace.generation.call_count == 2
        # Should log all followups as event
        mock_trace.event.assert_called_once()
        mock_client.flush.assert_called_once()

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_handles_exception_gracefully(self, mock_get_client):
        """Should handle exceptions and return False"""
        mock_client = Mock()
        mock_client.trace.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        result = log_followups(["Q?"], "query")
        assert result is False


class TestLogQueryResponse:
    """Tests for log_query_response() function"""

    def test_returns_false_when_disabled(self):
        """Should return False when Langfuse not configured"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            import utils.langfuse_logger as logger
            logger._langfuse_client = None

            result = log_query_response(
                query="test",
                response="answer",
                followups=[],
                context_sources=[]
            )
            assert result is False

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_logs_complete_interaction(self, mock_get_client):
        """Should log query, response, and context"""
        mock_client = Mock()
        mock_trace = Mock()
        mock_client.trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        result = log_query_response(
            query="What is PMF?",
            response="Product-market fit is...",
            followups=["How to measure?"],
            context_sources=[
                {"metadata": {"title": "Episode 1"}}
            ],
            session_id="test-session",
            latency_ms=150.5
        )

        assert result is True
        mock_client.trace.assert_called_once()
        # Should have span for retrieval
        mock_trace.span.assert_called()
        # Should have generation for LLM
        mock_trace.generation.assert_called()


class TestCreateSessionId:
    """Tests for create_session_id() function"""

    def test_returns_string(self):
        """Should return a string"""
        session_id = create_session_id()
        assert isinstance(session_id, str)

    def test_returns_uuid_format(self):
        """Should return valid UUID format"""
        session_id = create_session_id()
        # UUID format: 8-4-4-4-12 characters
        parts = session_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8

    def test_returns_unique_ids(self):
        """Should return unique IDs each time"""
        id1 = create_session_id()
        id2 = create_session_id()
        assert id1 != id2


class TestFlush:
    """Tests for flush() function"""

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_flushes_client(self, mock_get_client):
        """Should call flush on client"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        flush()

        mock_client.flush.assert_called_once()

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_handles_no_client(self, mock_get_client):
        """Should handle None client gracefully"""
        mock_get_client.return_value = None

        # Should not raise
        flush()

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_handles_flush_exception(self, mock_get_client):
        """Should handle flush exception gracefully"""
        mock_client = Mock()
        mock_client.flush.side_effect = Exception("Flush error")
        mock_get_client.return_value = mock_client

        # Should not raise
        flush()


class TestSkipSilently:
    """Tests verifying silent skip behavior"""

    def test_log_followups_skips_silently(self, capsys):
        """log_followups should skip silently when disabled"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            import utils.langfuse_logger as logger
            logger._langfuse_client = None

            result = log_followups(["Q?"], "query")

            # Should return False but not print anything to stdout
            assert result is False
            captured = capsys.readouterr()
            assert "error" not in captured.out.lower()
            assert "warning" not in captured.out.lower()
