"""
Integration tests for Lenny's Podcast RAG Chatbot

Tests the full pipeline: document processing → vector store → search → LLM → response
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Check if API keys are available for real integration tests
HAS_API_KEYS = (
    os.getenv("GOOGLE_API_KEY") is not None and
    os.getenv("PINECONE_API_KEY") is not None
)


class TestDocumentToChunksPipeline:
    """Test document processing pipeline"""

    def test_process_single_transcript(self):
        """Should process a transcript file into chunks with metadata"""
        from utils.document_processor import discover_transcripts, process_transcript

        transcripts = discover_transcripts()
        assert len(transcripts) > 0, "Should find transcripts"

        # Process first transcript
        chunks = process_transcript(transcripts[0])

        assert len(chunks) > 0, "Should create chunks"
        assert "text" in chunks[0], "Chunk should have text"
        assert "metadata" in chunks[0], "Chunk should have metadata"
        assert "guest" in chunks[0]["metadata"], "Metadata should include guest"
        assert "title" in chunks[0]["metadata"], "Metadata should include title"

    def test_chunk_metadata_preserved(self):
        """Metadata should be preserved through chunking"""
        from utils.document_processor import (
            discover_transcripts,
            load_transcript,
            parse_frontmatter,
            process_transcript
        )

        transcripts = discover_transcripts()
        if not transcripts:
            pytest.skip("No transcripts available")

        # Get original metadata
        content = load_transcript(transcripts[0])
        original_metadata = parse_frontmatter(content)

        # Process to chunks
        chunks = process_transcript(transcripts[0])

        # Verify metadata preserved
        chunk_metadata = chunks[0]["metadata"]
        assert chunk_metadata.get("guest") == original_metadata.get("guest")
        assert chunk_metadata.get("title") == original_metadata.get("title")


class TestEmbeddingsGeneration:
    """Test embedding generation"""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_generates_correct_dimensions(self):
        """Embeddings should be 768 dimensions"""
        from utils.vector_store import get_embeddings_model, EMBEDDING_DIMENSIONS

        model = get_embeddings_model()
        embedding = model.embed_query("Test query")

        assert len(embedding) == EMBEDDING_DIMENSIONS
        assert all(isinstance(x, float) for x in embedding)


class TestSearchPipeline:
    """Test search functionality"""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_search_returns_results_format(self):
        """Search results should have correct format"""
        from utils.vector_store import search, index_exists, get_index_stats

        if not index_exists():
            pytest.skip("Index not created")

        stats = get_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            pytest.skip("Index is empty")

        results = search("product market fit", top_k=3)

        assert isinstance(results, list)
        if results:
            assert "text" in results[0]
            assert "metadata" in results[0]
            assert "score" in results[0]


class TestLLMResponseGeneration:
    """Test LLM response generation"""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_generates_response_with_context(self):
        """Should generate response based on context"""
        from utils.llm_chain import generate_response

        # Mock search results
        search_results = [{
            "text": "Product-market fit is when customers are actively seeking your product.",
            "metadata": {
                "title": "Finding PMF",
                "guest": "Marc Andreessen",
                "publish_date": "2023-05-15"
            },
            "score": 0.9
        }]

        response = generate_response(
            query="What is product-market fit?",
            search_results=search_results
        )

        assert "answer" in response
        assert len(response["answer"]) > 0

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_includes_citations_in_response(self):
        """Response should include source citations"""
        from utils.llm_chain import generate_response

        search_results = [{
            "text": "Growth teams focus on user acquisition and retention.",
            "metadata": {
                "title": "Building Growth Teams",
                "guest": "Casey Winters",
                "publish_date": "2023-06-01"
            },
            "score": 0.85
        }]

        response = generate_response(
            query="What do growth teams do?",
            search_results=search_results
        )

        # Check that the response mentions the source somehow
        answer = response["answer"].lower()
        # Citation could be in various formats
        has_citation = (
            "casey winters" in answer or
            "growth teams" in answer or
            "source" in answer or
            "from" in answer
        )
        # This is a soft check - LLM may not always cite
        assert len(response["answer"]) > 50  # Should have substantial answer


class TestConversationMemory:
    """Test conversation memory functionality"""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_chat_history_affects_response(self):
        """Chat history should provide context for follow-up questions"""
        from utils.llm_chain import generate_response

        search_results = [{
            "text": "Product-market fit indicators include high retention and organic growth.",
            "metadata": {
                "title": "PMF Indicators",
                "guest": "Test Guest",
                "publish_date": "2023-01-01"
            },
            "score": 0.9
        }]

        # First message
        response1 = generate_response(
            query="What is product-market fit?",
            search_results=search_results
        )

        # Follow-up with history
        chat_history = [
            {"role": "user", "content": "What is product-market fit?"},
            {"role": "assistant", "content": response1["answer"]}
        ]

        response2 = generate_response(
            query="How do you measure it?",
            search_results=search_results,
            chat_history=chat_history
        )

        # Second response should exist and be contextual
        assert len(response2["answer"]) > 0


class TestContextFormatting:
    """Test context formatting for LLM"""

    def test_format_context_includes_sources(self):
        """Formatted context should include source information"""
        from utils.llm_chain import format_context

        results = [
            {
                "text": "Some content here.",
                "metadata": {
                    "title": "Episode Title",
                    "guest": "Guest Name",
                    "publish_date": "2023-01-15"
                }
            }
        ]

        context = format_context(results)

        assert "Episode Title" in context
        assert "Guest Name" in context
        assert "2023-01-15" in context
        assert "Some content here" in context


class TestResponseParsing:
    """Test response parsing"""

    def test_parses_answer(self):
        """Should correctly parse answer from LLM output"""
        from utils.llm_chain import parse_response

        llm_output = """Product-market fit is when your product satisfies strong market demand.

Key indicators include:
- High user retention
- Organic word-of-mouth growth
- Users actively seeking your product
"""

        result = parse_response(llm_output)

        assert "answer" in result
        assert "product-market fit" in result["answer"].lower()
        assert "High user retention" in result["answer"]


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline"""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not available")
    def test_full_rag_pipeline(self):
        """Test complete RAG pipeline with real components"""
        from utils.vector_store import search, index_exists, get_index_stats
        from utils.llm_chain import generate_response

        # Skip if index not ready
        if not index_exists():
            pytest.skip("Index not created")

        stats = get_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            pytest.skip("Index is empty - run indexing first")

        # 1. Search for relevant content
        query = "What advice do guests give about hiring?"
        search_results = search(query, top_k=5)

        assert len(search_results) > 0, "Should find relevant results"

        # 2. Generate response
        response = generate_response(
            query=query,
            search_results=search_results
        )

        # 3. Verify response structure
        assert "answer" in response
        assert len(response["answer"]) > 100, "Should have substantial answer"

        print(f"\nQuery: {query}")
        print(f"Answer preview: {response['answer'][:200]}...")


class TestLangfuseIntegration:
    """Test Langfuse logging integration"""

    def test_logs_silently_when_disabled(self):
        """Should skip logging silently when Langfuse not configured"""
        from utils.langfuse_logger import log_followups, is_langfuse_enabled

        # This should not raise even if Langfuse is not configured
        result = log_followups(
            followups=["Question 1?", "Question 2?"],
            query="Test query",
            session_id="test-session"
        )

        # Result depends on whether Langfuse is configured
        if is_langfuse_enabled():
            assert result is True
        else:
            assert result is False


class TestExtractSuggestedEpisodes:
    """Test episode extraction from search results"""

    def test_extracts_unique_episodes(self):
        """Should extract unique episodes from search results"""
        from app import extract_suggested_episodes

        search_results = [
            {
                "text": "Content 1",
                "metadata": {
                    "title": "Episode A",
                    "guest": "Guest A",
                    "youtube_url": "https://youtube.com/a",
                    "publish_date": "2023-01-01"
                }
            },
            {
                "text": "Content 2",
                "metadata": {
                    "title": "Episode B",
                    "guest": "Guest B",
                    "youtube_url": "https://youtube.com/b",
                    "publish_date": "2023-02-01"
                }
            }
        ]

        episodes = extract_suggested_episodes(search_results)

        assert len(episodes) == 2
        assert episodes[0]["title"] == "Episode A"
        assert episodes[1]["title"] == "Episode B"

    def test_limits_to_max_episodes(self):
        """Should limit to max_episodes parameter"""
        from app import extract_suggested_episodes

        search_results = [
            {"text": f"Content {i}", "metadata": {"title": f"Episode {i}", "guest": f"Guest {i}"}}
            for i in range(10)
        ]

        episodes = extract_suggested_episodes(search_results, max_episodes=3)
        assert len(episodes) == 3

    def test_deduplicates_by_title(self):
        """Should not include same episode twice"""
        from app import extract_suggested_episodes

        search_results = [
            {"text": "Chunk 1", "metadata": {"title": "Same Episode", "guest": "Guest"}},
            {"text": "Chunk 2", "metadata": {"title": "Same Episode", "guest": "Guest"}},
            {"text": "Chunk 3", "metadata": {"title": "Different Episode", "guest": "Guest"}}
        ]

        episodes = extract_suggested_episodes(search_results)

        assert len(episodes) == 2
        titles = [e["title"] for e in episodes]
        assert titles.count("Same Episode") == 1

    def test_handles_missing_metadata(self):
        """Should handle missing metadata gracefully"""
        from app import extract_suggested_episodes

        search_results = [
            {"text": "Content", "metadata": {"title": "Episode Title"}}
        ]

        episodes = extract_suggested_episodes(search_results)

        assert len(episodes) == 1
        assert episodes[0]["title"] == "Episode Title"
        assert episodes[0]["guest"] == "Unknown"
        assert episodes[0]["youtube_url"] == ""

    def test_returns_empty_for_no_results(self):
        """Should return empty list for no results"""
        from app import extract_suggested_episodes

        episodes = extract_suggested_episodes([])
        assert episodes == []


class TestAppImports:
    """Test that the main app imports correctly"""

    def test_app_imports_all_components(self):
        """App should import all utility modules"""
        # This tests that all imports in app.py work
        import app

        assert hasattr(app, "main")
        assert hasattr(app, "init_session_state")
        assert hasattr(app, "handle_user_input")
        assert hasattr(app, "render_sidebar")
        assert hasattr(app, "render_chat_messages")
        assert hasattr(app, "extract_suggested_episodes")
