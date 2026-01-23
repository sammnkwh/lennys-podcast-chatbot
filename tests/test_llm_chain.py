"""
Unit tests for LLM Chain (Component 3)

Uses mocking for most tests to avoid API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from utils.llm_chain import (
    get_llm,
    format_context,
    format_chat_history,
    build_prompt,
    parse_response,
    generate_response,
    test_llm_connection,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT
)


class TestGetLLM:
    """Tests for get_llm() function"""

    def test_raises_error_without_api_key(self):
        """Should raise ValueError if GOOGLE_API_KEY not set"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                get_llm()

    @patch("utils.llm_chain.ChatGoogleGenerativeAI")
    def test_returns_llm_instance(self, mock_chat):
        """Should return ChatGoogleGenerativeAI instance"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = get_llm()
            mock_chat.assert_called_once()

    @patch("utils.llm_chain.ChatGoogleGenerativeAI")
    def test_uses_default_model(self, mock_chat):
        """Should use default model when not specified"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            get_llm()
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("utils.llm_chain.ChatGoogleGenerativeAI")
    def test_accepts_custom_parameters(self, mock_chat):
        """Should accept custom temperature and max_tokens"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            get_llm(temperature=0.5, max_tokens=1000)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_output_tokens"] == 1000


class TestFormatContext:
    """Tests for format_context() function"""

    def test_returns_no_context_message_for_empty(self):
        """Should return message when no results"""
        result = format_context([])
        assert "No relevant context" in result

    def test_formats_single_result(self):
        """Should format single search result"""
        results = [{
            "text": "Product-market fit is important.",
            "metadata": {
                "title": "Finding PMF",
                "guest": "Marc Andreessen",
                "publish_date": "2023-05-15"
            }
        }]
        context = format_context(results)

        assert "Product-market fit is important" in context
        assert "Finding PMF" in context
        assert "Marc Andreessen" in context
        assert "2023-05-15" in context

    def test_formats_multiple_results(self):
        """Should format multiple results with separators"""
        results = [
            {
                "text": "First insight.",
                "metadata": {"title": "Episode 1", "guest": "Guest 1", "publish_date": "2023-01-01"}
            },
            {
                "text": "Second insight.",
                "metadata": {"title": "Episode 2", "guest": "Guest 2", "publish_date": "2023-02-01"}
            }
        ]
        context = format_context(results)

        assert "First insight" in context
        assert "Second insight" in context
        assert "[Source 1]" in context
        assert "[Source 2]" in context

    def test_handles_missing_metadata(self):
        """Should handle missing metadata gracefully"""
        results = [{
            "text": "Some text.",
            "metadata": {}
        }]
        context = format_context(results)

        assert "Some text" in context
        assert "Unknown" in context  # Default values


class TestFormatChatHistory:
    """Tests for format_chat_history() function"""

    def test_converts_user_messages(self):
        """Should convert user messages to HumanMessage"""
        from langchain_core.messages import HumanMessage

        messages = [{"role": "user", "content": "Hello"}]
        result = format_chat_history(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_converts_assistant_messages(self):
        """Should convert assistant messages to AIMessage"""
        from langchain_core.messages import AIMessage

        messages = [{"role": "assistant", "content": "Hi there!"}]
        result = format_chat_history(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)

    def test_handles_conversation(self):
        """Should handle alternating conversation"""
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"}
        ]
        result = format_chat_history(messages)

        assert len(result) == 3

    def test_handles_empty_history(self):
        """Should handle empty history"""
        result = format_chat_history([])
        assert result == []


class TestBuildPrompt:
    """Tests for build_prompt() function"""

    def test_includes_system_message(self):
        """Should include system message with context"""
        from langchain_core.messages import SystemMessage

        messages = build_prompt("Test query", "Test context")

        assert any(isinstance(m, SystemMessage) for m in messages)

    def test_includes_user_query(self):
        """Should include user query as last message"""
        from langchain_core.messages import HumanMessage

        messages = build_prompt("Test query", "Test context")

        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "Test query"

    def test_includes_context_in_system(self):
        """Should include context in system message"""
        messages = build_prompt("Query", "Important context here")

        system_msg = messages[0]
        assert "Important context here" in system_msg.content

    def test_includes_chat_history(self):
        """Should include chat history when provided"""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        messages = build_prompt("New query", "Context", chat_history=history)

        # Should have: system + 2 history messages + current query
        assert len(messages) == 4


class TestParseResponse:
    """Tests for parse_response() function"""

    def test_extracts_answer(self):
        """Should extract answer text"""
        response = """This is the answer.

FOLLOW-UP QUESTIONS:
- Question 1?
- Question 2?
"""
        result = parse_response(response)
        assert "This is the answer" in result["answer"]

    def test_extracts_followups(self):
        """Should extract follow-up questions"""
        response = """Answer here.

FOLLOW-UP QUESTIONS:
- How do you measure this?
- What are the common mistakes?
- When should you start?
"""
        result = parse_response(response)

        assert len(result["followups"]) >= 2
        assert any("measure" in q.lower() for q in result["followups"])

    def test_handles_no_followups(self):
        """Should handle response without follow-ups"""
        response = "Just an answer without follow-up questions."
        result = parse_response(response)

        assert result["answer"] == response
        assert result["followups"] == []

    def test_handles_different_formats(self):
        """Should handle different follow-up section formats"""
        formats = [
            "FOLLOW-UP QUESTIONS:\n- Q1?\n- Q2?",
            "Follow-up Questions:\n- Q1?\n- Q2?",
            "Suggested Questions:\n1. Q1?\n2. Q2?",
        ]

        for fmt in formats:
            response = f"Answer.\n\n{fmt}"
            result = parse_response(response)
            # Should find at least some questions
            assert "answer" in result
            assert "followups" in result

    def test_limits_followups_to_five(self):
        """Should limit follow-ups to 5"""
        response = """Answer.

FOLLOW-UP QUESTIONS:
- Question 1?
- Question 2?
- Question 3?
- Question 4?
- Question 5?
- Question 6?
- Question 7?
"""
        result = parse_response(response)
        assert len(result["followups"]) <= 5

    def test_returns_dict_format(self):
        """Should return dict with correct keys"""
        result = parse_response("Any response")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "followups" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["followups"], list)


class TestGenerateResponse:
    """Tests for generate_response() function"""

    @patch("utils.llm_chain.get_llm")
    def test_returns_correct_format(self, mock_get_llm):
        """Should return dict with answer, followups, raw_response"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Answer here.\n\nFOLLOW-UP QUESTIONS:\n- Q1?"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate_response(
            query="Test question",
            search_results=[{
                "text": "Context",
                "metadata": {"title": "T", "guest": "G", "publish_date": "2023"}
            }]
        )

        assert "answer" in result
        assert "followups" in result
        assert "raw_response" in result

    @patch("utils.llm_chain.get_llm")
    def test_passes_chat_history(self, mock_get_llm):
        """Should include chat history in prompt"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Response"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        history = [{"role": "user", "content": "Previous"}]
        generate_response(
            query="Current",
            search_results=[],
            chat_history=history
        )

        # Verify invoke was called with messages including history
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) >= 2  # System + history + query


class TestIntegration:
    """Integration tests requiring real API (skipped if unavailable)"""

    @pytest.fixture
    def has_api_key(self):
        """Check if API key is available"""
        return os.getenv("GOOGLE_API_KEY") is not None

    def test_real_llm_connection(self, has_api_key):
        """Test real LLM connection"""
        if not has_api_key:
            pytest.skip("API key not available")

        result = test_llm_connection()
        assert result is True

    def test_real_response_generation(self, has_api_key):
        """Test real response generation"""
        if not has_api_key:
            pytest.skip("API key not available")

        result = generate_response(
            query="What is product-market fit?",
            search_results=[{
                "text": "Product-market fit means customers love your product.",
                "metadata": {
                    "title": "Test Episode",
                    "guest": "Test Guest",
                    "publish_date": "2024-01-01"
                }
            }]
        )

        assert result["answer"]
        assert len(result["answer"]) > 10
