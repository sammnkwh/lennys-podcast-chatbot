"""
Langfuse Logger for Lenny Bot

Handles logging of query-response interactions to Langfuse
for LLM-as-a-judge evaluation and analytics.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langfuse client (lazy initialized)
_langfuse_client = None


def is_langfuse_enabled() -> bool:
    """
    Check if Langfuse is configured and enabled.

    Returns:
        True if all required Langfuse env vars are set
    """
    required_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY"
    ]
    return all(os.getenv(var) for var in required_vars)


def get_langfuse_client():
    """
    Get or create Langfuse client (lazy initialization).

    Returns:
        Langfuse client instance or None if not configured
    """
    global _langfuse_client

    if not is_langfuse_enabled():
        return None

    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
        except ImportError:
            print("Warning: langfuse package not installed")
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize Langfuse: {e}")
            return None

    return _langfuse_client


def log_followups(
    followups: List[str],
    query: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Log follow-up questions to Langfuse.

    Args:
        followups: List of follow-up question strings
        query: The user query that generated these follow-ups
        session_id: Optional session ID for grouping
        metadata: Optional additional metadata

    Returns:
        True if logged successfully, False otherwise
    """
    if not followups:
        return True  # Nothing to log

    client = get_langfuse_client()
    if client is None:
        # Langfuse not configured - skip silently
        return False

    try:
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        trace_id = client.create_trace_id()
        trace_context = {"trace_id": trace_id}

        # Create the main span for followups
        with client.start_as_current_span(
            name="followup_questions",
            trace_context=trace_context,
            input=query,
            metadata={
                "source": "Lenny Bot",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "followup_count": len(followups),
                **(metadata or {})
            }
        ) as span:
            # Set session_id on the trace
            client.update_current_trace(session_id=session_id)

            # Log each follow-up as a generation
            for i, followup in enumerate(followups):
                with client.start_as_current_generation(
                    name=f"followup_{i + 1}",
                    input=query,
                    output=followup,
                    metadata={
                        "followup_index": i + 1,
                        "total_followups": len(followups)
                    }
                ):
                    pass

            # Update span with combined output
            span.update(output="\n".join(f"- {q}" for q in followups))

        # Flush to ensure data is sent
        client.flush()

        return True

    except Exception as e:
        print(f"Warning: Failed to log to Langfuse: {e}")
        return False


def log_query_response(
    query: str,
    response: str,
    suggested_episodes: List[Dict[str, Any]],
    context_sources: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    latency_ms: Optional[float] = None
) -> bool:
    """
    Log a complete query-response interaction to Langfuse.

    Each chunk and suggested episode is logged as individual fields
    for easy export to spreadsheets for LLM-as-a-judge evaluation.

    Args:
        query: User's question
        response: Assistant's answer
        suggested_episodes: List of suggested episode dicts with title, guest, youtube_url
        context_sources: List of search results with text, score, and metadata
        session_id: Optional session ID
        latency_ms: Optional response latency in milliseconds

    Returns:
        True if logged successfully
    """
    client = get_langfuse_client()
    if client is None:
        return False

    try:
        if session_id is None:
            session_id = str(uuid.uuid4())

        trace_id = client.create_trace_id()
        trace_context = {"trace_id": trace_id}

        # Build metadata with individual chunk fields
        metadata = {
            "source": "Lenny Bot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suggested_episode_count": len(suggested_episodes),
            "context_count": len(context_sources),
            "latency_ms": latency_ms
        }

        # Add individual chunk fields (chunk_1_text, chunk_1_source, chunk_1_score, etc.)
        for i, chunk in enumerate(context_sources):
            chunk_num = i + 1
            chunk_metadata = chunk.get("metadata", {})

            # Get chunk text
            chunk_text = chunk.get("text", "")

            # Build source string: "Episode Title | Guest Name"
            title = chunk_metadata.get("title", "Unknown")
            guest = chunk_metadata.get("guest", "")
            chunk_source = f"{title} | {guest}" if guest else title

            # Get similarity score
            chunk_score = chunk.get("score", 0.0)

            metadata[f"chunk_{chunk_num}_text"] = chunk_text
            metadata[f"chunk_{chunk_num}_source"] = chunk_source
            metadata[f"chunk_{chunk_num}_score"] = round(chunk_score, 4) if chunk_score else 0.0

        # Add individual suggested episode fields
        for i, episode in enumerate(suggested_episodes):
            ep_num = i + 1
            metadata[f"suggested_{ep_num}_title"] = episode.get("title", "")
            metadata[f"suggested_{ep_num}_guest"] = episode.get("guest", "")
            metadata[f"suggested_{ep_num}_url"] = episode.get("youtube_url", "")

        # Create the main trace span
        with client.start_as_current_span(
            name="rag_query",
            trace_context=trace_context,
            input=query,
            output=response,
            metadata=metadata
        ):
            # Set session_id on the trace
            client.update_current_trace(session_id=session_id)

            # Log the retrieval step
            source_titles = [s.get("metadata", {}).get("title", "Unknown") for s in context_sources]
            with client.start_as_current_span(
                name="retrieval",
                input=query,
                output=f"Retrieved {len(context_sources)} sources: {', '.join(source_titles)}"
            ):
                pass

            # Log the generation step
            with client.start_as_current_generation(
                name="llm_response",
                input=query,
                output=response,
                metadata={"model": "gemini-2.5-flash"}
            ):
                pass

            # Log suggested episodes as an event
            if suggested_episodes:
                episode_lines = [
                    f"- {ep.get('title', 'Unknown')} ({ep.get('guest', 'Unknown')})"
                    for ep in suggested_episodes
                ]
                client.create_event(
                    name="suggested_episodes",
                    output="\n".join(episode_lines)
                )

        client.flush()
        return True

    except Exception as e:
        print(f"Warning: Failed to log to Langfuse: {e}")
        return False


def create_session_id() -> str:
    """
    Create a new session ID for tracking conversations.

    Returns:
        UUID string for session tracking
    """
    return str(uuid.uuid4())


def flush():
    """
    Flush any pending Langfuse events.
    Call this before the app shuts down.
    """
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick test
    print("Testing Langfuse logger...")

    print(f"\n1. Langfuse enabled: {is_langfuse_enabled()}")

    if is_langfuse_enabled():
        print("\n2. Testing log_query_response...")
        result = log_query_response(
            query="What is product-market fit?",
            response="Product-market fit is when your product meets a strong market demand.",
            suggested_episodes=[
                {"title": "How Superhuman Built an Engine", "guest": "Rahul Vohra", "youtube_url": "https://youtube.com/example"}
            ],
            context_sources=[
                {"text": "PMF is when customers love your product...", "score": 0.89, "metadata": {"title": "Superhuman", "guest": "Rahul Vohra"}}
            ],
            session_id="test-session",
            latency_ms=150.5
        )
        print(f"   Logged: {result}")
    else:
        print("\n2. Skipping log test - Langfuse not configured")
        print("   To enable, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
