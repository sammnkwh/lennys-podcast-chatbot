"""
Lenny's Podcast RAG Chatbot

A Streamlit app that answers questions about Lenny's Podcast episodes
using RAG (Retrieval-Augmented Generation).
"""

from __future__ import annotations  # PEP 604 unions (int | None) on Python 3.9

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import utility modules
from utils.document_processor import get_transcript_count, process_all_transcripts
from utils.vector_store import (
    search,
    store_chunks,
    get_index_stats,
    index_exists,
    INDEX_NAME
)
from utils.llm_chain import (
    generate_response,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE
)
from utils.langfuse_logger import (
    create_session_id,
    is_langfuse_enabled,
    log_query_response
)
import time

# Page configuration
st.set_page_config(
    page_title="Lenny's Podcast Q&A",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Follow-up button styling */
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }

    /* Suggestion buttons - secondary style */
    .followup-btn > button {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        color: #31333F;
    }

    .followup-btn > button:hover {
        background-color: #e0e2e6;
        border-color: #c0c2c6;
    }

    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "suggested_episodes" not in st.session_state:
        st.session_state.suggested_episodes = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_id()

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "indexing" not in st.session_state:
        st.session_state.indexing = False


def clear_chat():
    """Clear chat history and start new conversation."""
    st.session_state.messages = []
    st.session_state.suggested_episodes = []
    st.session_state.session_id = create_session_id()


def render_sidebar():
    """Render the sidebar with settings and controls."""
    with st.sidebar:
        st.title("🎙️ Lenny's Podcast Q&A")
        st.caption("Ask questions about podcast episodes")

        st.divider()

        # New Chat button
        if st.button("🔄 New Chat", use_container_width=True):
            clear_chat()
            st.rerun()

        st.divider()

        # Index status
        st.subheader("📊 Index Status")

        try:
            transcript_count = get_transcript_count()
            st.metric("Available Transcripts", transcript_count)
        except Exception:
            transcript_count = 0
            st.metric("Available Transcripts", "Unknown")

        if index_exists():
            stats = get_index_stats()
            vector_count = stats.get("total_vector_count", 0)
            st.metric("Indexed Chunks", vector_count)

            if vector_count == 0:
                st.warning("Index is empty. Click below to index transcripts.")
        else:
            st.warning("Index not created yet.")
            vector_count = 0

        # Index button
        if transcript_count > 0:
            if st.button(
                "📥 Index Transcripts" if vector_count == 0 else "🔄 Re-index Transcripts",
                use_container_width=True,
                disabled=st.session_state.indexing
            ):
                index_transcripts()

        st.divider()

        # Settings section
        with st.expander("⚙️ Settings", expanded=False):
            # Temperature slider
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher values make responses more creative, lower values more focused."
            )

            # System prompt
            st.text_area(
                "System Prompt",
                value=st.session_state.system_prompt,
                height=200,
                key="system_prompt_input",
                on_change=lambda: setattr(
                    st.session_state,
                    "system_prompt",
                    st.session_state.system_prompt_input
                ),
                help="Customize how the assistant responds."
            )

            # Reset prompt button
            if st.button("Reset to Default", use_container_width=True):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.rerun()

        st.divider()

        # Info section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            This chatbot answers questions based on transcripts from
            **Lenny's Podcast**.

            **Features:**
            - Semantic search across all episodes
            - Source citations with episode info
            - Suggested episodes to explore further

            **Model:** Gemini 2.5 Flash
            """)

            if is_langfuse_enabled():
                st.success("📊 Langfuse logging enabled")
            else:
                st.info("📊 Langfuse logging disabled")


def index_transcripts():
    """Index all transcripts to Pinecone."""
    st.session_state.indexing = True

    with st.sidebar:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Loading transcripts...")

            def progress_callback(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Processing: {current}/{total} transcripts")

            # Process all transcripts
            chunks = process_all_transcripts(progress_callback=progress_callback)

            status_text.text(f"Indexing {len(chunks)} chunks...")
            progress_bar.progress(0)

            def index_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Indexing: {current}/{total} chunks")

            # Store in Pinecone
            stored = store_chunks(chunks, progress_callback=index_progress)

            progress_bar.progress(1.0)
            status_text.text(f"✓ Indexed {stored} chunks successfully!")
            st.success(f"Indexed {stored} chunks from {get_transcript_count()} transcripts!")

        except Exception as e:
            st.error(f"Error indexing: {str(e)}")
        finally:
            st.session_state.indexing = False


def parse_duration_to_seconds(duration_str: str) -> int | None:
    """
    Parse a duration string like '1:54:40' or '54:40' into total seconds.

    Returns:
        Total seconds, or None if parsing fails
    """
    try:
        parts = duration_str.strip().split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return None
    except (ValueError, AttributeError):
        return None


def format_timestamp(total_seconds: int) -> str:
    """Format seconds as H:MM:SS or MM:SS."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def extract_suggested_episodes(search_results: list, max_episodes: int = 3) -> list:
    """
    Extract unique episodes from search results.

    Args:
        search_results: List of search results with metadata
        max_episodes: Maximum number of episodes to return

    Returns:
        List of episode dicts with title, guest, youtube_url, publish_date, estimated_timestamp
    """
    seen_titles = set()
    episodes = []

    for result in search_results:
        metadata = result.get("metadata", {})
        title = metadata.get("title")

        if title and title not in seen_titles:
            seen_titles.add(title)

            # Compute estimated timestamp
            estimated_timestamp = ""
            chunk_id = metadata.get("chunk_id")
            chunk_total = metadata.get("chunk_total")
            duration_seconds = metadata.get("duration_seconds")
            duration_str = metadata.get("duration", "")

            if chunk_id is not None and chunk_total and chunk_total > 0:
                total_secs = duration_seconds or parse_duration_to_seconds(duration_str)
                if total_secs:
                    start_secs = int((chunk_id / chunk_total) * total_secs)
                    estimated_timestamp = f"~{format_timestamp(start_secs)}"

            episodes.append({
                "title": title,
                "guest": metadata.get("guest", "Unknown"),
                "youtube_url": metadata.get("youtube_url", ""),
                "publish_date": metadata.get("publish_date", ""),
                "estimated_timestamp": estimated_timestamp
            })

        if len(episodes) >= max_episodes:
            break

    return episodes


def render_chat_messages():
    """Render chat message history."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show suggested episodes after the last assistant message
            if (message["role"] == "assistant" and
                i == len(st.session_state.messages) - 1 and
                st.session_state.suggested_episodes):
                render_suggested_episodes()


def render_suggested_episodes():
    """Render suggested episode cards."""
    if not st.session_state.suggested_episodes:
        return

    st.markdown("---")
    st.caption("🎧 Suggested episodes to explore:")

    for idx, episode in enumerate(st.session_state.suggested_episodes):
        title = episode.get("title", "Unknown Episode")
        guest = episode.get("guest", "Unknown")
        youtube_url = episode.get("youtube_url", "")

        timestamp = episode.get("estimated_timestamp", "")
        timestamp_label = f" ({timestamp})" if timestamp else ""

        # Create a clickable episode card
        if youtube_url:
            st.markdown(f"**[{title}]({youtube_url})**{timestamp_label}  \n_{guest}_")
        else:
            st.markdown(f"**{title}**{timestamp_label}  \n_{guest}_")


def handle_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Clear previous suggested episodes
    st.session_state.suggested_episodes = []

    # Check if index has data
    if not index_exists():
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ The search index hasn't been created yet. Please click **Index Transcripts** in the sidebar first."
        })
        return

    stats = get_index_stats()
    if stats.get("total_vector_count", 0) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ The search index is empty. Please click **Index Transcripts** in the sidebar to index the podcast transcripts."
        })
        return

    try:
        # Search for relevant context
        search_results = search(user_input, top_k=5)

        # Prepare chat history for context
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]  # Exclude current message
        ][-10:]  # Keep last 10 messages for context

        # Generate response with timing
        start_time = time.time()
        response = generate_response(
            query=user_input,
            search_results=search_results,
            chat_history=chat_history,
            system_prompt=st.session_state.system_prompt,
            temperature=st.session_state.temperature
        )
        latency_ms = (time.time() - start_time) * 1000

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"]
        })

        # Extract suggested episodes from search results
        st.session_state.suggested_episodes = extract_suggested_episodes(search_results)

        # Log to Langfuse for evaluation
        log_query_response(
            query=user_input,
            response=response["answer"],
            suggested_episodes=st.session_state.suggested_episodes,
            context_sources=search_results,
            session_id=st.session_state.session_id,
            latency_ms=latency_ms
        )

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Sorry, I encountered an error: {str(e)}"
        })


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.header("💬 Chat")

    # Render chat messages
    render_chat_messages()

    # Chat input
    if prompt := st.chat_input("Ask a question about Lenny's Podcast..."):
        handle_user_input(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
