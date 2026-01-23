"""
Langfuse Logger for Lenny's Podcast RAG Chatbot

Handles optional logging of follow-up questions to Langfuse
for evaluation and analytics.
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

        # Create a trace for this interaction
        trace = client.trace(
            name="followup_questions",
            session_id=session_id,
            metadata={
                "source": "lenny-podcast-rag",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }
        )

        # Log each follow-up as a generation
        for i, followup in enumerate(followups):
            trace.generation(
                name=f"followup_{i + 1}",
                input=query,
                output=followup,
                metadata={
                    "followup_index": i + 1,
                    "total_followups": len(followups)
                }
            )

        # Also log all follow-ups together as an event
        trace.event(
            name="all_followups",
            input=query,
            output="\n".join(f"- {q}" for q in followups),
            metadata={
                "followup_count": len(followups)
            }
        )

        # Flush to ensure data is sent
        client.flush()

        return True

    except Exception as e:
        print(f"Warning: Failed to log to Langfuse: {e}")
        return False


def log_query_response(
    query: str,
    response: str,
    followups: List[str],
    context_sources: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    latency_ms: Optional[float] = None
) -> bool:
    """
    Log a complete query-response interaction to Langfuse.

    Args:
        query: User's question
        response: Assistant's answer
        followups: Generated follow-up questions
        context_sources: List of sources used (with metadata)
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

        # Create trace for the full interaction
        trace = client.trace(
            name="rag_query",
            session_id=session_id,
            input=query,
            output=response,
            metadata={
                "source": "lenny-podcast-rag",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "followup_count": len(followups),
                "context_count": len(context_sources),
                "latency_ms": latency_ms
            }
        )

        # Log the retrieval step
        source_titles = [s.get("metadata", {}).get("title", "Unknown") for s in context_sources]
        trace.span(
            name="retrieval",
            input=query,
            output=f"Retrieved {len(context_sources)} sources: {', '.join(source_titles)}"
        )

        # Log the generation step
        trace.generation(
            name="llm_response",
            input=query,
            output=response,
            metadata={
                "model": "gemini-2.5-flash"
            }
        )

        # Log follow-ups
        if followups:
            trace.event(
                name="followup_questions",
                output="\n".join(f"- {q}" for q in followups)
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
        print("\n2. Testing log_followups...")
        result = log_followups(
            followups=[
                "What is product-market fit?",
                "How do you measure PMF?"
            ],
            query="Tell me about growth",
            session_id="test-session"
        )
        print(f"   Logged: {result}")
    else:
        print("\n2. Skipping log test - Langfuse not configured")
        print("   To enable, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
