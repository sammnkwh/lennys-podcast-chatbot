"""
Vector Store for Lenny's Podcast RAG Chatbot

Handles Pinecone connection, index management, embedding generation,
and similarity search.
"""

import os
import time
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "lennys-podcast"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSIONS = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 100  # Pinecone upsert batch size


def get_pinecone_client() -> Pinecone:
    """
    Initialize and return Pinecone client.

    Returns:
        Pinecone client instance

    Raises:
        ValueError: If PINECONE_API_KEY is not set
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    return Pinecone(api_key=api_key)


def get_or_create_index(
    index_name: str = INDEX_NAME,
    dimensions: int = EMBEDDING_DIMENSIONS
) -> Any:
    """
    Get existing Pinecone index or create a new one.

    Args:
        index_name: Name of the index
        dimensions: Vector dimensions (768 for text-embedding-004)

    Returns:
        Pinecone Index object
    """
    pc = get_pinecone_client()

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        # Create new serverless index
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    return pc.Index(index_name)


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Get Google's text-embedding-004 model.

    Returns:
        GoogleGenerativeAIEmbeddings instance

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


def generate_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Generate a unique ID for a chunk based on its content and metadata.

    Args:
        chunk: Chunk dict with 'text' and 'metadata'

    Returns:
        Unique string ID
    """
    # Create a hash from guest_slug, chunk_id, and a portion of text
    metadata = chunk.get("metadata", {})
    guest_slug = metadata.get("guest_slug", "unknown")
    chunk_id = metadata.get("chunk_id", 0)
    text_sample = chunk.get("text", "")[:100]

    unique_string = f"{guest_slug}_{chunk_id}_{text_sample}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for Pinecone (flatten and convert types).

    Pinecone metadata values must be strings, numbers, booleans, or lists of strings.

    Args:
        metadata: Raw metadata dict

    Returns:
        Cleaned metadata dict safe for Pinecone
    """
    cleaned = {}

    # Fields to keep for citations and filtering
    keep_fields = [
        "guest", "title", "publish_date", "guest_slug",
        "chunk_id", "chunk_total", "duration", "youtube_url"
    ]

    for key in keep_fields:
        if key in metadata:
            value = metadata[key]
            # Convert dates to strings
            if hasattr(value, 'isoformat'):
                cleaned[key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert list items to strings
                cleaned[key] = [str(v) for v in value]
            else:
                cleaned[key] = str(value)

    return cleaned


def store_chunks(
    chunks: List[Dict[str, Any]],
    index_name: str = INDEX_NAME,
    progress_callback: Optional[callable] = None
) -> int:
    """
    Embed and store chunks in Pinecone.

    Args:
        chunks: List of chunks with 'text' and 'metadata'
        index_name: Pinecone index name
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Number of chunks stored
    """
    if not chunks:
        return 0

    index = get_or_create_index(index_name)
    embeddings_model = get_embeddings_model()

    total_stored = 0

    # Process in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings
        embeddings = embeddings_model.embed_documents(texts)

        # Prepare vectors for upsert
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vector_id = generate_chunk_id(chunk)
            metadata = prepare_metadata(chunk["metadata"])
            # Store the text in metadata for retrieval
            metadata["text"] = chunk["text"]

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        total_stored += len(vectors)

        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, len(chunks)), len(chunks))

    return total_stored


def search(
    query: str,
    top_k: int = 5,
    index_name: str = INDEX_NAME,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks using similarity search.

    Args:
        query: Search query string
        top_k: Number of results to return
        index_name: Pinecone index name
        filter_dict: Optional metadata filter

    Returns:
        List of results with 'text', 'metadata', and 'score'
    """
    index = get_or_create_index(index_name)
    embeddings_model = get_embeddings_model()

    # Generate query embedding
    query_embedding = embeddings_model.embed_query(query)

    # Search Pinecone
    search_kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }

    if filter_dict:
        search_kwargs["filter"] = filter_dict

    results = index.query(**search_kwargs)

    # Format results
    formatted_results = []
    for match in results.matches:
        metadata = dict(match.metadata)
        text = metadata.pop("text", "")

        formatted_results.append({
            "text": text,
            "metadata": metadata,
            "score": match.score
        })

    return formatted_results


def get_index_stats(index_name: str = INDEX_NAME) -> Dict[str, Any]:
    """
    Get statistics about the Pinecone index.

    Args:
        index_name: Pinecone index name

    Returns:
        Dict with index statistics
    """
    try:
        index = get_or_create_index(index_name)
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {}
        }
    except Exception as e:
        return {"error": str(e)}


def delete_all_vectors(index_name: str = INDEX_NAME) -> bool:
    """
    Delete all vectors from the index.

    Args:
        index_name: Pinecone index name

    Returns:
        True if successful
    """
    try:
        index = get_or_create_index(index_name)
        index.delete(delete_all=True)
        return True
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        return False


def index_exists(index_name: str = INDEX_NAME) -> bool:
    """
    Check if the Pinecone index exists.

    Args:
        index_name: Pinecone index name

    Returns:
        True if index exists
    """
    try:
        pc = get_pinecone_client()
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        return index_name in existing_indexes
    except Exception:
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing vector store...")

    # Check if we can connect
    try:
        pc = get_pinecone_client()
        print("✓ Pinecone client connected")

        indexes = [idx.name for idx in pc.list_indexes()]
        print(f"✓ Existing indexes: {indexes}")

        if index_exists():
            stats = get_index_stats()
            print(f"✓ Index stats: {stats}")
    except Exception as e:
        print(f"✗ Error: {e}")
