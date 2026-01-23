# Project Handoff: Lenny's Podcast RAG Chatbot

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on transcripts from Lenny's Podcast. The chatbot generates follow-up questions after each response and logs them to Langfuse for evaluation.

## Data Source

- **Repository:** `sammnkwh/lennys-podcast-transcripts`
- **Content:** 269 podcast episode transcripts
- **Format:** Markdown files with YAML frontmatter
- **Structure:**
  ```
  episodes/
    guest-name/
      transcript.md
  ```
- **Metadata available in frontmatter:**
  - `guest` - Guest name
  - `title` - Episode title
  - `publish_date` - Publication date
  - `youtube_url` - YouTube link
  - `duration` - Episode length
  - `keywords` - Topic tags
  - `description` - Episode summary

## Scope Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector DB | Pinecone only | Free tier, fast, cloud-based |
| Document format | MD files | Transcripts are already in MD |
| Source citations | Yes | Cite episode title, guest, date |
| Document removal | Not needed | Add-only for now |
| Access | Personal use | No auth required |
| Langfuse logging | Follow-ups only | User will configure eval in Langfuse |
| Chat history | Ephemeral | Stored in `st.session_state`, clears on refresh |
| Eval dashboard | Not building | User will use Langfuse instead |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain (Python) |
| UI | Streamlit |
| LLM | Google Gemini (1.5 Flash or 2.0 Flash) |
| Embeddings | Google text-embedding-004 |
| Vector DB | Pinecone (free tier) |
| Logging | Langfuse (optional) |
| Deployment | Streamlit Community Cloud (future) |

## MECE Components

| # | Component | Responsibility | File |
|---|-----------|----------------|------|
| 1 | Document Processor | Load MD files, parse YAML, chunk text | `utils/document_processor.py` |
| 2 | Vector Store | Pinecone index, embeddings, search | `utils/vector_store.py` |
| 3 | LLM Chain | RAG chain, response parsing, citations | `utils/llm_chain.py` |
| 4 | Langfuse Logger | Log follow-up questions | `utils/langfuse_logger.py` |
| 5 | Streamlit UI | Chat interface, follow-up buttons | `app.py` |

## File Structure

```
Billable Bot (2)/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored)
├── .env.example                # Template for env vars
├── .gitignore                  # Git ignore rules
├── HANDOFF.md                  # This document
├── rag_chatbot_spec.md         # Original spec (reference)
├── generate-tasks.md           # Task generation framework
├── API_keys.txt                # Personal key backup (gitignored)
├── utils/
│   ├── __init__.py
│   ├── document_processor.py   # Component 1
│   ├── vector_store.py         # Component 2
│   ├── llm_chain.py            # Component 3
│   └── langfuse_logger.py      # Component 4
├── data/
│   └── transcripts/            # Cloned transcript repo
└── tasks/
    └── tasks-lenny-rag-chatbot.md  # Task checklist
```

## Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional (for Langfuse logging)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Note:** Real keys are stored in `.env` (gitignored). User has both Gemini and Pinecone keys ready.

## Key Technical Details

### Document Processing
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Splitter: LangChain's `RecursiveCharacterTextSplitter`
- Metadata preserved: guest, title, date, chunk_id

### Vector Store
- Pinecone index: auto-created on first run
- Dimensions: 768 (text-embedding-004 output)
- Top-k retrieval: 5 chunks

### LLM Chain
- Model: Gemini 1.5 Flash or 2.0 Flash
- System prompt: Customizable in UI
- Output format: Answer + 3-5 follow-up questions
- Citations: "From '[title]' with [guest] ([date])"

### UI Features
- Sidebar: New Chat button, settings, system prompt editor
- Chat: Message history, loading spinner
- Follow-ups: Clickable buttons after each response
- State: Ephemeral (session-based)

## GitHub Repository

- **Repo:** `sammnkwh/Billable-Bot-2`
- **URL:** https://github.com/sammnkwh/Billable-Bot-2
- **Branch strategy:** Feature branches, merge to main

## Current Status

- [x] Project setup (git, .env, .gitignore)
- [x] MECE component breakdown
- [x] Task list generated (54 sub-tasks)
- [ ] Implementation (not started)

## Next Steps

1. Create feature branch: `git checkout -b feature/lenny-rag-chatbot`
2. Follow tasks in `tasks/tasks-lenny-rag-chatbot.md`
3. Check off tasks as completed
4. Commit and push after each component

## Questions Resolved

1. **Vector DB choice?** → Pinecone only (no ChromaDB toggle needed)
2. **Document format?** → MD with YAML frontmatter
3. **Need eval dashboard?** → No, using Langfuse instead
4. **What to log to Langfuse?** → Just follow-up questions
5. **Chat history persistence?** → Ephemeral (session state only)
6. **Source citations?** → Yes, include episode/guest/date

## Open Questions

None currently. Ready to implement.
