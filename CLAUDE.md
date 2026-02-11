# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG chatbot that answers questions about Lenny's Podcast using 303 episode transcripts. Built with Streamlit, LangChain, Gemini, and Pinecone.

## Commands

```bash
# Run the app
python3 -m streamlit run app.py

# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_vector_store.py -v

# Run a single test class or method
python3 -m pytest tests/test_vector_store.py::TestPrepareMetadata -v
python3 -m pytest tests/test_vector_store.py::TestPrepareMetadata::test_keeps_allowed_fields -v

# Export Langfuse traces to CSV
python3 scripts/export_traces.py --days 7 --output traces_export.csv
```

## Architecture

The app follows a 5-component pipeline:

1. **Document Processor** (`utils/document_processor.py`) — Loads transcript `.md` files from `data/transcripts/episodes/{guest-name}/transcript.md`, parses YAML frontmatter, and chunks text (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`.

2. **Vector Store** (`utils/vector_store.py`) — Manages Pinecone index (`lennys-podcast`), embedding generation, and similarity search. Uses `gemini-embedding-001` with 768 dimensions. **Critical:** uses `task_type="retrieval_document"` for indexing and `task_type="retrieval_query"` for search — these must not be mixed. Has rate-limit retry logic for the Google embedding API.

3. **LLM Chain** (`utils/llm_chain.py`) — Builds RAG prompts with retrieved context and chat history, calls Gemini 2.5 Flash, and parses responses. Context is formatted with source citations.

4. **Langfuse Logger** (`utils/langfuse_logger.py`) — Optional observability. Logs queries, responses, retrieved chunks, and suggested episodes as structured metadata. Disabled gracefully when keys are missing.

5. **Streamlit UI** (`app.py`) — Chat interface with sidebar controls. After each response, extracts suggested episodes from search results (with estimated timestamps based on chunk position) and renders them as clickable YouTube links.

### Data flow

```
User query → embed (retrieval_query) → Pinecone search (top 5)
  → format context with citations → Gemini LLM → response
  → extract suggested episodes from search results → render UI
  → log to Langfuse
```

### Re-indexing

Only needed when transcripts change, chunking strategy changes, or embedding model/settings change. Run via the sidebar "Index Transcripts" button or programmatically:

```python
from utils.vector_store import store_chunks, delete_all_vectors
from utils.document_processor import process_all_transcripts
delete_all_vectors()
chunks = process_all_transcripts()
store_chunks(chunks)  # ~37K chunks, handles rate limits automatically
```

## Environment Variables

Required in `.env`:
- `GOOGLE_API_KEY` — Gemini API (paid tier needed for indexing volume)
- `PINECONE_API_KEY`

Optional:
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Transcript Metadata

Each transcript's YAML frontmatter includes: `guest`, `title`, `youtube_url`, `video_id`, `publish_date`, `description`, `duration_seconds`, `duration`, `view_count`, `channel`, `keywords`. The fields stored in Pinecone are filtered by `keep_fields` in `prepare_metadata()`.

## Testing

Tests use mocking to avoid real API calls. Integration tests that need real keys are skipped automatically when keys are missing. The test suite has 113 tests covering all components.
