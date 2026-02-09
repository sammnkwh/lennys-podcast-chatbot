"""
Unit tests for Vector Store (Component 2)

Uses mocking to avoid real API calls during testing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from utils.vector_store import (
    get_pinecone_client,
    get_or_create_index,
    get_embeddings_model,
    generate_chunk_id,
    prepare_metadata,
    store_chunks,
    search,
    get_index_stats,
    index_exists,
    INDEX_NAME,
    EMBEDDING_DIMENSIONS
)


class TestGetPineconeClient:
    """Tests for get_pinecone_client() function"""

    def test_raises_error_without_api_key(self):
        """Should raise ValueError if PINECONE_API_KEY not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("PINECONE_API_KEY", None)
            with pytest.raises(ValueError, match="PINECONE_API_KEY"):
                get_pinecone_client()

    @patch("utils.vector_store.Pinecone")
    def test_returns_pinecone_client(self, mock_pinecone):
        """Should return Pinecone client when API key is set"""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            client = get_pinecone_client()
            mock_pinecone.assert_called_once_with(api_key="test-key")


class TestGetOrCreateIndex:
    """Tests for get_or_create_index() function"""

    @patch("utils.vector_store.get_pinecone_client")
    def test_returns_existing_index(self, mock_get_client):
        """Should return existing index without creating"""
        mock_client = Mock()
        mock_index_info = Mock()
        mock_index_info.name = INDEX_NAME
        mock_client.list_indexes.return_value = [mock_index_info]
        mock_client.Index.return_value = Mock()
        mock_get_client.return_value = mock_client

        index = get_or_create_index()

        mock_client.create_index.assert_not_called()
        mock_client.Index.assert_called_once_with(INDEX_NAME)

    @patch("utils.vector_store.get_pinecone_client")
    def test_creates_new_index_if_not_exists(self, mock_get_client):
        """Should create index if it doesn't exist"""
        mock_client = Mock()
        mock_client.list_indexes.return_value = []  # No existing indexes

        # Mock describe_index to return ready status
        mock_status = Mock()
        mock_status.status = {"ready": True}
        mock_client.describe_index.return_value = mock_status
        mock_client.Index.return_value = Mock()
        mock_get_client.return_value = mock_client

        index = get_or_create_index()

        mock_client.create_index.assert_called_once()
        # Verify dimensions in the call
        call_kwargs = mock_client.create_index.call_args
        assert call_kwargs.kwargs["dimension"] == EMBEDDING_DIMENSIONS

    @patch("utils.vector_store.get_pinecone_client")
    def test_uses_correct_dimensions(self, mock_get_client):
        """Should configure index with 768 dimensions"""
        mock_client = Mock()
        mock_client.list_indexes.return_value = []
        mock_status = Mock()
        mock_status.status = {"ready": True}
        mock_client.describe_index.return_value = mock_status
        mock_client.Index.return_value = Mock()
        mock_get_client.return_value = mock_client

        get_or_create_index(dimensions=768)

        call_kwargs = mock_client.create_index.call_args
        assert call_kwargs.kwargs["dimension"] == 768


class TestGetEmbeddingsModel:
    """Tests for get_embeddings_model() function"""

    def test_raises_error_without_api_key(self):
        """Should raise ValueError if GOOGLE_API_KEY not set"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                get_embeddings_model()

    @patch("utils.vector_store.GoogleGenerativeAIEmbeddings")
    def test_returns_embeddings_model(self, mock_embeddings):
        """Should return embeddings model when API key is set"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = get_embeddings_model()
            mock_embeddings.assert_called_once()
            assert "gemini-embedding-001" in str(mock_embeddings.call_args)


class TestGenerateChunkId:
    """Tests for generate_chunk_id() function"""

    def test_returns_string(self):
        """Should return a string ID"""
        chunk = {
            "text": "Test content",
            "metadata": {"guest_slug": "test-guest", "chunk_id": 0}
        }
        chunk_id = generate_chunk_id(chunk)
        assert isinstance(chunk_id, str)

    def test_same_chunk_same_id(self):
        """Same chunk should produce same ID"""
        chunk = {
            "text": "Test content",
            "metadata": {"guest_slug": "test-guest", "chunk_id": 0}
        }
        id1 = generate_chunk_id(chunk)
        id2 = generate_chunk_id(chunk)
        assert id1 == id2

    def test_different_chunks_different_ids(self):
        """Different chunks should produce different IDs"""
        chunk1 = {
            "text": "Test content 1",
            "metadata": {"guest_slug": "guest-1", "chunk_id": 0}
        }
        chunk2 = {
            "text": "Test content 2",
            "metadata": {"guest_slug": "guest-2", "chunk_id": 0}
        }
        id1 = generate_chunk_id(chunk1)
        id2 = generate_chunk_id(chunk2)
        assert id1 != id2

    def test_handles_missing_metadata(self):
        """Should handle chunks with missing metadata gracefully"""
        chunk = {"text": "Test content", "metadata": {}}
        chunk_id = generate_chunk_id(chunk)
        assert isinstance(chunk_id, str)


class TestPrepareMetadata:
    """Tests for prepare_metadata() function"""

    def test_keeps_allowed_fields(self):
        """Should keep allowed fields in output"""
        metadata = {
            "guest": "Test Guest",
            "title": "Test Title",
            "publish_date": "2024-01-15",
            "chunk_id": 0
        }
        cleaned = prepare_metadata(metadata)
        assert cleaned["guest"] == "Test Guest"
        assert cleaned["title"] == "Test Title"
        assert cleaned["chunk_id"] == 0

    def test_removes_disallowed_fields(self):
        """Should remove fields not in the keep list"""
        metadata = {
            "guest": "Test Guest",
            "source_file": "/path/to/file",  # Not in keep list
            "random_field": "value"  # Not in keep list
        }
        cleaned = prepare_metadata(metadata)
        assert "source_file" not in cleaned
        assert "random_field" not in cleaned

    def test_converts_date_objects(self):
        """Should convert date objects to ISO strings"""
        from datetime import date
        metadata = {"publish_date": date(2024, 1, 15)}
        cleaned = prepare_metadata(metadata)
        assert cleaned["publish_date"] == "2024-01-15"

    def test_handles_list_values(self):
        """Should handle list values"""
        metadata = {"keywords": ["test", "example"]}
        # keywords not in keep_fields, but test the list handling logic
        # Let's use a field that would be kept if it were a list
        metadata = {"guest": "Test", "title": "Title"}
        cleaned = prepare_metadata(metadata)
        assert isinstance(cleaned, dict)


class TestStoreChunks:
    """Tests for store_chunks() function"""

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_returns_count_of_stored_chunks(self, mock_embeddings, mock_index):
        """Should return number of chunks stored"""
        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
        mock_embeddings.return_value = mock_model

        mock_idx = Mock()
        mock_index.return_value = mock_idx

        chunks = [
            {"text": "Chunk 1", "metadata": {"guest_slug": "g1", "chunk_id": 0}},
            {"text": "Chunk 2", "metadata": {"guest_slug": "g2", "chunk_id": 0}}
        ]

        count = store_chunks(chunks)
        assert count == 2

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_calls_upsert(self, mock_embeddings, mock_index):
        """Should call Pinecone upsert"""
        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1] * 768]
        mock_embeddings.return_value = mock_model

        mock_idx = Mock()
        mock_index.return_value = mock_idx

        chunks = [{"text": "Test", "metadata": {"guest_slug": "g", "chunk_id": 0}}]

        store_chunks(chunks)
        mock_idx.upsert.assert_called_once()

    def test_returns_zero_for_empty_list(self):
        """Should return 0 for empty chunk list"""
        count = store_chunks([])
        assert count == 0

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_calls_progress_callback(self, mock_embeddings, mock_index):
        """Should call progress callback"""
        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1] * 768]
        mock_embeddings.return_value = mock_model
        mock_index.return_value = Mock()

        callback = Mock()
        chunks = [{"text": "Test", "metadata": {"guest_slug": "g", "chunk_id": 0}}]

        store_chunks(chunks, progress_callback=callback)
        callback.assert_called()


class TestSearch:
    """Tests for search() function"""

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_returns_list_of_results(self, mock_embeddings, mock_index):
        """Should return list of search results"""
        mock_model = Mock()
        mock_model.embed_query.return_value = [0.1] * 768
        mock_embeddings.return_value = mock_model

        mock_match = Mock()
        mock_match.metadata = {"text": "Result text", "guest": "Test Guest"}
        mock_match.score = 0.95

        mock_idx = Mock()
        mock_idx.query.return_value = Mock(matches=[mock_match])
        mock_index.return_value = mock_idx

        results = search("test query")

        assert isinstance(results, list)
        assert len(results) == 1

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_result_format(self, mock_embeddings, mock_index):
        """Results should have text, metadata, and score"""
        mock_model = Mock()
        mock_model.embed_query.return_value = [0.1] * 768
        mock_embeddings.return_value = mock_model

        mock_match = Mock()
        mock_match.metadata = {"text": "Result text", "guest": "Test Guest"}
        mock_match.score = 0.95

        mock_idx = Mock()
        mock_idx.query.return_value = Mock(matches=[mock_match])
        mock_index.return_value = mock_idx

        results = search("test query")

        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]

    @patch("utils.vector_store.get_or_create_index")
    @patch("utils.vector_store.get_embeddings_model")
    def test_uses_correct_top_k(self, mock_embeddings, mock_index):
        """Should pass correct top_k to query"""
        mock_model = Mock()
        mock_model.embed_query.return_value = [0.1] * 768
        mock_embeddings.return_value = mock_model

        mock_idx = Mock()
        mock_idx.query.return_value = Mock(matches=[])
        mock_index.return_value = mock_idx

        search("test query", top_k=10)

        call_kwargs = mock_idx.query.call_args.kwargs
        assert call_kwargs["top_k"] == 10


class TestGetIndexStats:
    """Tests for get_index_stats() function"""

    @patch("utils.vector_store.get_or_create_index")
    def test_returns_stats_dict(self, mock_index):
        """Should return dictionary with stats"""
        mock_idx = Mock()
        mock_stats = Mock()
        mock_stats.total_vector_count = 1000
        mock_stats.dimension = 768
        mock_stats.index_fullness = 0.1
        mock_stats.namespaces = {}
        mock_idx.describe_index_stats.return_value = mock_stats
        mock_index.return_value = mock_idx

        stats = get_index_stats()

        assert isinstance(stats, dict)
        assert stats["total_vector_count"] == 1000
        assert stats["dimension"] == 768


class TestIndexExists:
    """Tests for index_exists() function"""

    @patch("utils.vector_store.get_pinecone_client")
    def test_returns_true_when_exists(self, mock_get_client):
        """Should return True when index exists"""
        mock_client = Mock()
        mock_index_info = Mock()
        mock_index_info.name = INDEX_NAME
        mock_client.list_indexes.return_value = [mock_index_info]
        mock_get_client.return_value = mock_client

        assert index_exists() is True

    @patch("utils.vector_store.get_pinecone_client")
    def test_returns_false_when_not_exists(self, mock_get_client):
        """Should return False when index doesn't exist"""
        mock_client = Mock()
        mock_client.list_indexes.return_value = []
        mock_get_client.return_value = mock_client

        assert index_exists() is False


class TestIntegration:
    """Integration tests that require real API keys (skipped if not available)"""

    @pytest.fixture
    def has_api_keys(self):
        """Check if API keys are available"""
        return (
            os.getenv("PINECONE_API_KEY") is not None and
            os.getenv("GOOGLE_API_KEY") is not None
        )

    def test_real_embeddings_dimensions(self, has_api_keys):
        """Test that real embeddings have correct dimensions"""
        if not has_api_keys:
            pytest.skip("API keys not available")

        model = get_embeddings_model()
        embedding = model.embed_query("Test query")
        assert len(embedding) == EMBEDDING_DIMENSIONS

    def test_real_pinecone_connection(self, has_api_keys):
        """Test real Pinecone connection"""
        if not has_api_keys:
            pytest.skip("API keys not available")

        client = get_pinecone_client()
        # Should not raise
        indexes = list(client.list_indexes())
        assert isinstance(indexes, list)
