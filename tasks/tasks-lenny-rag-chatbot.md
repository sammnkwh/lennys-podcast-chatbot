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

### Notes

- No tests for now - focus on getting the chatbot working first
- Transcripts will be loaded from cloned GitHub repo (sammnkwh/lennys-podcast-transcripts)

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.0.1 Read file` → `- [x] 1.0.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [ ] 0.0 Create feature branch
  - [ ] 0.0.1 Create and checkout a new branch (`git checkout -b feature/lenny-rag-chatbot`)

### Component 1: Document Processor

- [ ] 1.0 Set up MD file loading from transcript repo
  - [ ] 1.0.1 Clone or copy the lennys-podcast-transcripts repo into `data/transcripts/`
  - [ ] 1.0.2 Create `utils/__init__.py` file
  - [ ] 1.0.3 Create `utils/document_processor.py` with function to discover all transcript MD files
  - [ ] 1.0.4 Implement `load_transcript(file_path)` function that reads MD file content

- [ ] 1.1 Implement YAML frontmatter parsing
  - [ ] 1.1.1 Add `python-frontmatter` or `pyyaml` to requirements.txt
  - [ ] 1.1.2 Implement `parse_frontmatter(content)` to extract metadata (guest, title, date, etc.)
  - [ ] 1.1.3 Implement `get_transcript_body(content)` to extract text after frontmatter

- [ ] 1.2 Implement text chunking with metadata preservation
  - [ ] 1.2.1 Use LangChain's RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
  - [ ] 1.2.2 Implement `chunk_transcript(body, metadata)` that attaches metadata to each chunk
  - [ ] 1.2.3 Implement `process_all_transcripts()` that returns all chunks with metadata
  - [ ] 1.2.4 Test with a single transcript to verify chunking works correctly

### Component 2: Vector Store

- [ ] 2.0 Set up Pinecone connection and auto-create index
  - [ ] 2.0.1 Add `pinecone-client` to requirements.txt
  - [ ] 2.0.2 Create `utils/vector_store.py` with Pinecone initialization function
  - [ ] 2.0.3 Implement `get_or_create_index(index_name)` that creates index if it doesn't exist
  - [ ] 2.0.4 Configure index for 768 dimensions (text-embedding-004 output size)

- [ ] 2.1 Implement embedding generation and storage
  - [ ] 2.1.1 Add `langchain-google-genai` to requirements.txt
  - [ ] 2.1.2 Implement `get_embeddings_model()` using Google's text-embedding-004
  - [ ] 2.1.3 Implement `store_chunks(chunks)` that embeds and upserts chunks to Pinecone
  - [ ] 2.1.4 Include metadata (guest, title, date, chunk_id) in Pinecone vectors

- [ ] 2.2 Implement similarity search with metadata retrieval
  - [ ] 2.2.1 Implement `search(query, top_k=5)` that returns relevant chunks
  - [ ] 2.2.2 Ensure returned results include full metadata for citations
  - [ ] 2.2.3 Test search with sample query to verify retrieval works

### Component 3: LLM Chain

- [ ] 3.0 Set up Gemini LLM connection
  - [ ] 3.0.1 Implement `get_llm()` function using Gemini 1.5 Flash or 2.0 Flash
  - [ ] 3.0.2 Configure temperature and max tokens parameters
  - [ ] 3.0.3 Test basic LLM call to verify connection works

- [ ] 3.1 Build RAG chain with conversation memory
  - [ ] 3.1.1 Create `utils/llm_chain.py` with main chain logic
  - [ ] 3.1.2 Create system prompt template for Lenny's Podcast Q&A
  - [ ] 3.1.3 Implement prompt that includes: system prompt, context chunks, chat history, user question
  - [ ] 3.1.4 Add instruction to generate 3-5 follow-up questions at the end of each response

- [ ] 3.2 Implement response parsing (answer + follow-ups)
  - [ ] 3.2.1 Implement `parse_response(llm_output)` to separate answer from follow-up questions
  - [ ] 3.2.2 Handle case where LLM doesn't follow format (graceful fallback)
  - [ ] 3.2.3 Return structured dict: `{"answer": str, "followups": list[str]}`

- [ ] 3.3 Implement citation formatting from metadata
  - [ ] 3.3.1 Format context chunks to include source info in prompt
  - [ ] 3.3.2 Instruct LLM to cite sources as "From '[title]' with [guest] ([date])"
  - [ ] 3.3.3 Test that citations appear correctly in responses

### Component 4: Langfuse Logger

- [ ] 4.0 Set up Langfuse client connection
  - [ ] 4.0.1 Add `langfuse` to requirements.txt
  - [ ] 4.0.2 Create `utils/langfuse_logger.py` with initialization function
  - [ ] 4.0.3 Add Langfuse env vars to `.env.example` (commented out)
  - [ ] 4.0.4 Implement `is_langfuse_enabled()` check for optional usage

- [ ] 4.1 Implement follow-up question logging
  - [ ] 4.1.1 Implement `log_followups(followups, query, session_id)` function
  - [ ] 4.1.2 Include relevant metadata (timestamp, user query that generated them)
  - [ ] 4.1.3 Handle case when Langfuse is not configured (skip silently)

### Component 5: Streamlit UI

- [ ] 5.0 Set up Streamlit app structure and config
  - [ ] 5.0.1 Add `streamlit` to requirements.txt
  - [ ] 5.0.2 Create `app.py` with basic Streamlit page config
  - [ ] 5.0.3 Set page title, icon, and layout (wide)
  - [ ] 5.0.4 Add app title and description header

- [ ] 5.1 Build sidebar (settings, new chat button)
  - [ ] 5.1.1 Add "New Chat" button that clears conversation history
  - [ ] 5.1.2 Add expandable "Settings" section with temperature slider
  - [ ] 5.1.3 Add system prompt text area (editable, with default prompt)
  - [ ] 5.1.4 Display count of indexed transcripts

- [ ] 5.2 Build chat interface with message history
  - [ ] 5.2.1 Display chat messages using `st.chat_message()` for user and assistant
  - [ ] 5.2.2 Add chat input box at bottom using `st.chat_input()`
  - [ ] 5.2.3 Show loading spinner while generating response

- [ ] 5.3 Implement follow-up question buttons
  - [ ] 5.3.1 After each assistant message, display follow-up questions as clickable buttons
  - [ ] 5.3.2 Use `st.button()` in columns for clean layout
  - [ ] 5.3.3 Clicking a button submits that question as the next user input
  - [ ] 5.3.4 Style buttons to look like suggestions (not primary actions)

- [ ] 5.4 Implement session state management
  - [ ] 5.4.1 Initialize `st.session_state.messages` for chat history
  - [ ] 5.4.2 Initialize `st.session_state.followups` for current follow-up questions
  - [ ] 5.4.3 Handle state reset on "New Chat" button click
  - [ ] 5.4.4 Persist system prompt edits in session state

### Component 6: Integration

- [ ] 6.0 Wire all components together in app.py
  - [ ] 6.0.1 Import all utility modules
  - [ ] 6.0.2 On app start, check if Pinecone index exists and has vectors
  - [ ] 6.0.3 Wire user input → vector search → LLM chain → display response → log to Langfuse
  - [ ] 6.0.4 Add one-time "Index Transcripts" button for initial embedding

- [ ] 6.1 End-to-end testing with sample queries
  - [ ] 6.1.1 Test: "What does Lenny say about product-market fit?"
  - [ ] 6.1.2 Test: "Who has talked about growth teams?"
  - [ ] 6.1.3 Verify citations appear correctly
  - [ ] 6.1.4 Verify follow-up buttons work
  - [ ] 6.1.5 Verify conversation memory works across turns

- [ ] 6.2 Final commit and push
  - [ ] 6.2.1 Review all files for any hardcoded secrets (should be none)
  - [ ] 6.2.2 Update requirements.txt with final dependencies
  - [ ] 6.2.3 Commit all changes with descriptive message
  - [ ] 6.2.4 Push to GitHub
