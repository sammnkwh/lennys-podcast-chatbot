# Tasks: Lenny's Podcast RAG Chatbot

## MECE Components

| # | Component | Responsibility | Input | Output | Files |
|---|-----------|----------------|-------|--------|-------|
| 1 | Document Processor | Load MD transcripts, parse YAML frontmatter, chunk text with metadata | File paths | List of chunks with metadata (guest, title, date) | `utils/document_processor.py` |
| 2 | Vector Store | Create Pinecone index, store embeddings, similarity search | Chunks (to store) or query (to search) | Confirmation or relevant chunks with metadata | `utils/vector_store.py` |
| 3 | LLM Chain | Build prompt with context + history, call Gemini, parse answer + follow-ups | Query, context, chat history | Answer (with citations) + follow-up questions | `utils/llm_chain.py` |
| 4 | Langfuse Logger | Send follow-up questions to Langfuse | Follow-up questions, metadata | Logged event | `utils/langfuse_logger.py` |
| 5 | Streamlit UI | Sidebar, chat interface, follow-up buttons, session state | User interactions | Rendered app | `app.py` |

## Relevant Files

### Component 1: Document Processor
- `utils/document_processor.py` - Load MD files, parse YAML frontmatter, chunk text with metadata

### Component 2: Vector Store
- `utils/vector_store.py` - Pinecone connection, index management, embedding storage and retrieval

### Component 3: LLM Chain
- `utils/llm_chain.py` - Gemini LLM setup, RAG chain, response parsing, citation formatting

### Component 4: Langfuse Logger
- `utils/langfuse_logger.py` - Langfuse client setup and follow-up logging

### Component 5: Streamlit UI
- `app.py` - Main Streamlit application

### Supporting Files
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys) - gitignored
- `.env.example` - Template for environment variables
- `data/transcripts/` - Local copy of transcript files

### Test Files
- `tests/test_document_processor.py` - Unit tests for Component 1
- `tests/test_vector_store.py` - Unit tests for Component 2
- `tests/test_llm_chain.py` - Unit tests for Component 3
- `tests/test_langfuse_logger.py` - Unit tests for Component 4
- `tests/test_integration.py` - Integration tests for Component 6

### Notes

- Unit tests added after each component
- Transcripts will be loaded from cloned GitHub repo (sammnkwh/lennys-podcast-transcripts)

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.0.1 Read file` → `- [x] 1.0.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.0.1 Create and checkout a new branch (`git checkout -b feature/lenny-rag-chatbot`)

### Component 1: Document Processor

- [x] 1.0 Set up MD file loading from transcript repo
  - [x] 1.0.1 Clone or copy the lennys-podcast-transcripts repo into `data/transcripts/`
  - [x] 1.0.2 Create `utils/__init__.py` file
  - [x] 1.0.3 Create `utils/document_processor.py` with function to discover all transcript MD files
  - [x] 1.0.4 Implement `load_transcript(file_path)` function that reads MD file content

- [x] 1.1 Implement YAML frontmatter parsing
  - [x] 1.1.1 Add `python-frontmatter` or `pyyaml` to requirements.txt
  - [x] 1.1.2 Implement `parse_frontmatter(content)` to extract metadata (guest, title, date, etc.)
  - [x] 1.1.3 Implement `get_transcript_body(content)` to extract text after frontmatter

- [x] 1.2 Implement text chunking with metadata preservation
  - [x] 1.2.1 Use LangChain's RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  - [x] 1.2.2 Implement `chunk_transcript(body, metadata)` that attaches metadata to each chunk
  - [x] 1.2.3 Implement `process_all_transcripts()` that returns all chunks with metadata
  - [x] 1.2.4 Test with a single transcript to verify chunking works correctly

- [x] 1.3 Unit tests for Document Processor
  - [x] 1.3.1 Add `pytest` to requirements.txt
  - [x] 1.3.2 Create `tests/__init__.py` and `tests/test_document_processor.py`
  - [x] 1.3.3 Test `discover_transcripts()` finds MD files
  - [x] 1.3.4 Test `parse_frontmatter()` extracts metadata correctly
  - [x] 1.3.5 Test `chunk_transcript()` creates chunks with metadata
  - [x] 1.3.6 Run tests and verify all pass (28 tests passed)

### Component 2: Vector Store

- [x] 2.0 Set up Pinecone connection and auto-create index
  - [x] 2.0.1 Add `pinecone` to requirements.txt (renamed from pinecone-client)
  - [x] 2.0.2 Create `utils/vector_store.py` with Pinecone initialization function
  - [x] 2.0.3 Implement `get_or_create_index(index_name)` that creates index if it doesn't exist
  - [x] 2.0.4 Configure index for 768 dimensions (text-embedding-004 output size)

- [x] 2.1 Implement embedding generation and storage
  - [x] 2.1.1 Add `langchain-google-genai` to requirements.txt
  - [x] 2.1.2 Implement `get_embeddings_model()` using Google's text-embedding-004
  - [x] 2.1.3 Implement `store_chunks(chunks)` that embeds and upserts chunks to Pinecone
  - [x] 2.1.4 Include metadata (guest, title, date, chunk_id) in Pinecone vectors

- [x] 2.2 Implement similarity search with metadata retrieval
  - [x] 2.2.1 Implement `search(query, top_k=5)` that returns relevant chunks
  - [x] 2.2.2 Ensure returned results include full metadata for citations
  - [ ] 2.2.3 Test search with sample query to verify retrieval works (need indexed data)

- [x] 2.3 Unit tests for Vector Store
  - [x] 2.3.1 Create `tests/test_vector_store.py`
  - [x] 2.3.2 Test `get_or_create_index()` creates index with correct dimensions
  - [x] 2.3.3 Test `get_embeddings_model()` returns valid embeddings
  - [x] 2.3.4 Test `store_chunks()` and `search()` round-trip (mock Pinecone for unit tests)
  - [x] 2.3.5 Run tests and verify all pass (27 tests passed)

### Component 3: LLM Chain

- [x] 3.0 Set up Gemini LLM connection
  - [x] 3.0.1 Implement `get_llm()` function using Gemini 2.5 Flash
  - [x] 3.0.2 Configure temperature and max tokens parameters
  - [x] 3.0.3 Test basic LLM call to verify connection works

- [x] 3.1 Build RAG chain with conversation memory
  - [x] 3.1.1 Create `utils/llm_chain.py` with main chain logic
  - [x] 3.1.2 Create system prompt template for Lenny's Podcast Q&A
  - [x] 3.1.3 Implement prompt that includes: system prompt, context chunks, chat history, user question
  - [x] 3.1.4 Add instruction to generate 3-5 follow-up questions at the end of each response

- [x] 3.2 Implement response parsing (answer + follow-ups)
  - [x] 3.2.1 Implement `parse_response(llm_output)` to separate answer from follow-up questions
  - [x] 3.2.2 Handle case where LLM doesn't follow format (graceful fallback)
  - [x] 3.2.3 Return structured dict: `{"answer": str, "followups": list[str]}`

- [x] 3.3 Implement citation formatting from metadata
  - [x] 3.3.1 Format context chunks to include source info in prompt
  - [x] 3.3.2 Instruct LLM to cite sources as "From '[title]' with [guest] ([date])"
  - [x] 3.3.3 Test that citations appear correctly in responses

- [x] 3.4 Unit tests for LLM Chain
  - [x] 3.4.1 Create `tests/test_llm_chain.py`
  - [x] 3.4.2 Test `parse_response()` correctly separates answer from follow-ups
  - [x] 3.4.3 Test `parse_response()` graceful fallback when format is wrong
  - [x] 3.4.4 Test prompt building includes context and history
  - [x] 3.4.5 Run tests and verify all pass (27 tests passed)

### Component 4: Langfuse Logger

- [x] 4.0 Set up Langfuse client connection
  - [x] 4.0.1 Add `langfuse` to requirements.txt
  - [x] 4.0.2 Create `utils/langfuse_logger.py` with initialization function
  - [x] 4.0.3 Add Langfuse env vars to `.env.example` (commented out)
  - [x] 4.0.4 Implement `is_langfuse_enabled()` check for optional usage

- [x] 4.1 Implement follow-up question logging
  - [x] 4.1.1 Implement `log_followups(followups, query, session_id)` function
  - [x] 4.1.2 Include relevant metadata (timestamp, user query that generated them)
  - [x] 4.1.3 Handle case when Langfuse is not configured (skip silently)

- [x] 4.2 Unit tests for Langfuse Logger
  - [x] 4.2.1 Create `tests/test_langfuse_logger.py`
  - [x] 4.2.2 Test `is_langfuse_enabled()` returns False when env vars missing
  - [x] 4.2.3 Test `log_followups()` skips silently when Langfuse disabled
  - [x] 4.2.4 Run tests and verify all pass (19 tests passed)

### Component 5: Streamlit UI

- [x] 5.0 Set up Streamlit app structure and config
  - [x] 5.0.1 Add `streamlit` to requirements.txt
  - [x] 5.0.2 Create `app.py` with basic Streamlit page config
  - [x] 5.0.3 Set page title, icon, and layout (wide)
  - [x] 5.0.4 Add app title and description header

- [x] 5.1 Build sidebar (settings, new chat button)
  - [x] 5.1.1 Add "New Chat" button that clears conversation history
  - [x] 5.1.2 Add expandable "Settings" section with temperature slider
  - [x] 5.1.3 Add system prompt text area (editable, with default prompt)
  - [x] 5.1.4 Display count of indexed transcripts

- [x] 5.2 Build chat interface with message history
  - [x] 5.2.1 Display chat messages using `st.chat_message()` for user and assistant
  - [x] 5.2.2 Add chat input box at bottom using `st.chat_input()`
  - [x] 5.2.3 Show loading spinner while generating response

- [x] 5.3 Implement follow-up question buttons
  - [x] 5.3.1 After each assistant message, display follow-up questions as clickable buttons
  - [x] 5.3.2 Use `st.button()` in columns for clean layout
  - [x] 5.3.3 Clicking a button submits that question as the next user input
  - [x] 5.3.4 Style buttons to look like suggestions (not primary actions)

- [x] 5.4 Implement session state management
  - [x] 5.4.1 Initialize `st.session_state.messages` for chat history
  - [x] 5.4.2 Initialize `st.session_state.followups` for current follow-up questions
  - [x] 5.4.3 Handle state reset on "New Chat" button click
  - [x] 5.4.4 Persist system prompt edits in session state

### Component 6: Integration

- [x] 6.0 Wire all components together in app.py
  - [x] 6.0.1 Import all utility modules
  - [x] 6.0.2 On app start, check if Pinecone index exists and has vectors
  - [x] 6.0.3 Wire user input → vector search → LLM chain → display response → log to Langfuse
  - [x] 6.0.4 Add one-time "Index Transcripts" button for initial embedding

- [x] 6.1 Integration tests
  - [x] 6.1.1 Create `tests/test_integration.py`
  - [x] 6.1.2 Test full pipeline: document → vector store → search → LLM → response
  - [x] 6.1.3 Test conversation memory persists across turns
  - [x] 6.1.4 Run integration tests and verify all pass (107 passed, 6 skipped)

- [x] 6.2 End-to-end manual testing with sample queries
  - [x] 6.2.1 Test: "What does Lenny say about product-market fit?"
  - [x] 6.2.2 Test: "Who has talked about growth teams?"
  - [x] 6.2.3 Verify citations appear correctly
  - [x] 6.2.4 Verify follow-up buttons work
  - [x] 6.2.5 Verify conversation memory works across turns

- [x] 6.3 Final commit and push
  - [x] 6.3.1 Review all files for any hardcoded secrets (should be none)
  - [x] 6.3.2 Update requirements.txt with final dependencies
  - [x] 6.3.3 Commit all changes with descriptive message
  - [x] 6.3.4 Push to GitHub
