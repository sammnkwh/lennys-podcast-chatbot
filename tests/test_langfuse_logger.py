"""
Unit tests for Langfuse Logger (Component 4)

Uses mocking to avoid real Langfuse API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
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


def create_mock_langfuse_client():
    """Helper to create a properly mocked Langfuse v3 client"""
    mock_client = Mock()
    mock_client.create_trace_id.return_value = "test-trace-id"

    # Create context manager mocks for spans and generations
    @contextmanager
    def mock_span(*args, **kwargs):
        span = Mock()
        span.update = Mock()
        yield span

    @contextmanager
    def mock_generation(*args, **kwargs):
        yield Mock()

    mock_client.start_as_current_span = Mock(side_effect=mock_span)
    mock_client.start_as_current_generation = Mock(side_effect=mock_generation)
    mock_client.update_current_trace = Mock()
    mock_client.create_event = Mock()
    mock_client.flush = Mock()

    return mock_client


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
    def test_creates_span_with_followups(self, mock_get_client):
        """Should create span and log follow-ups"""
        mock_client = create_mock_langfuse_client()
        mock_get_client.return_value = mock_client

        result = log_followups(
            followups=["Question 1?", "Question 2?"],
            query="Original query",
            session_id="test-session"
        )

        assert result is True
        mock_client.create_trace_id.assert_called_once()
        mock_client.start_as_current_span.assert_called_once()
        # Should log each followup as generation
        assert mock_client.start_as_current_generation.call_count == 2
        mock_client.flush.assert_called_once()

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_handles_exception_gracefully(self, mock_get_client):
        """Should handle exceptions and return False"""
        mock_client = Mock()
        mock_client.create_trace_id.side_effect = Exception("API error")
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
                suggested_episodes=[],
                context_sources=[]
            )
            assert result is False

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_logs_complete_interaction(self, mock_get_client):
        """Should log query, response, and context"""
        mock_client = create_mock_langfuse_client()
        mock_get_client.return_value = mock_client

        result = log_query_response(
            query="What is PMF?",
            response="Product-market fit is...",
            suggested_episodes=[
                {"title": "Superhuman Episode", "guest": "Rahul Vohra", "youtube_url": "https://youtube.com/example"}
            ],
            context_sources=[
                {"text": "PMF content here...", "score": 0.89, "metadata": {"title": "Episode 1", "guest": "Guest 1"}}
            ],
            session_id="test-session",
            latency_ms=150.5
        )

        assert result is True
        mock_client.create_trace_id.assert_called_once()
        # Main span + retrieval span = 2 calls
        assert mock_client.start_as_current_span.call_count == 2
        # LLM generation
        mock_client.start_as_current_generation.assert_called_once()
        # Event for suggested episodes
        mock_client.create_event.assert_called_once()

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_logs_individual_chunk_fields(self, mock_get_client):
        """Should log each chunk as individual fields for spreadsheet export"""
        mock_client = create_mock_langfuse_client()
        mock_get_client.return_value = mock_client

        context_sources = [
            {"text": "First chunk text", "score": 0.95, "metadata": {"title": "Episode A", "guest": "Guest A"}},
            {"text": "Second chunk text", "score": 0.87, "metadata": {"title": "Episode B", "guest": "Guest B"}},
        ]

        log_query_response(
            query="Test query",
            response="Test response",
            suggested_episodes=[],
            context_sources=context_sources,
            session_id="test-session"
        )

        # Check that start_as_current_span was called with individual chunk fields in metadata
        call_kwargs = mock_client.start_as_current_span.call_args_list[0][1]
        metadata = call_kwargs["metadata"]

        assert metadata["chunk_1_text"] == "First chunk text"
        assert metadata["chunk_1_source"] == "Episode A | Guest A"
        assert metadata["chunk_1_score"] == 0.95

        assert metadata["chunk_2_text"] == "Second chunk text"
        assert metadata["chunk_2_source"] == "Episode B | Guest B"
        assert metadata["chunk_2_score"] == 0.87

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_logs_individual_episode_fields(self, mock_get_client):
        """Should log each suggested episode as individual fields"""
        mock_client = create_mock_langfuse_client()
        mock_get_client.return_value = mock_client

        suggested_episodes = [
            {"title": "How Superhuman Built", "guest": "Rahul Vohra", "youtube_url": "https://youtube.com/ep1"},
            {"title": "Growth Tactics", "guest": "Brian Balfour", "youtube_url": "https://youtube.com/ep2"},
        ]

        log_query_response(
            query="Test query",
            response="Test response",
            suggested_episodes=suggested_episodes,
            context_sources=[],
            session_id="test-session"
        )

        # Check that start_as_current_span was called with individual episode fields in metadata
        call_kwargs = mock_client.start_as_current_span.call_args_list[0][1]
        metadata = call_kwargs["metadata"]

        assert metadata["suggested_1_title"] == "How Superhuman Built"
        assert metadata["suggested_1_guest"] == "Rahul Vohra"
        assert metadata["suggested_1_url"] == "https://youtube.com/ep1"

        assert metadata["suggested_2_title"] == "Growth Tactics"
        assert metadata["suggested_2_guest"] == "Brian Balfour"
        assert metadata["suggested_2_url"] == "https://youtube.com/ep2"

    @patch("utils.langfuse_logger.get_langfuse_client")
    def test_handles_missing_metadata_gracefully(self, mock_get_client):
        """Should handle chunks with missing metadata fields"""
        mock_client = create_mock_langfuse_client()
        mock_get_client.return_value = mock_client

        # Chunk with minimal metadata
        context_sources = [
            {"text": "Some text", "metadata": {"title": "Episode Only"}},  # No guest, no score
        ]

        log_query_response(
            query="Test query",
            response="Test response",
            suggested_episodes=[],
            context_sources=context_sources,
            session_id="test-session"
        )

        call_kwargs = mock_client.start_as_current_span.call_args_list[0][1]
        metadata = call_kwargs["metadata"]

        assert metadata["chunk_1_text"] == "Some text"
        assert metadata["chunk_1_source"] == "Episode Only"  # No guest, so just title
        assert metadata["chunk_1_score"] == 0.0  # Default score


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
