# Lenny's Podcast Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that lets you ask questions across 303 episodes of [Lenny's Podcast](https://www.lennysnewsletter.com/podcast). Built with Streamlit, LangChain, Google Gemini, and Pinecone.

## What It Does

Ask natural language questions like:
- *"What do guests say about finding product-market fit?"*
- *"Which episodes cover pricing strategies?"*
- *"What has Lenny's guests said about building in public?"*

The chatbot searches across all episode transcripts, retrieves the most relevant passages, and generates an answer with citations and suggested episode links (with estimated timestamps).

## Architecture

The app follows a 5-step pipeline:

```
User query â†’ embed (retrieval_query) â†’ Pinecone search (top 5)
  â†’ format context with citations â†’ Gemini LLM â†’ response
  â†’ extract suggested episodes â†’ render UI â†’ log to Langfuse
```

| Component | File | Description |
|-----------|------|-------------|
| Document Processor | `utils/document_processor.py` | Loads transcript `.md` files, parses YAML frontmatter, chunks text (1000 chars, 200 overlap) |
| Vector Store | `utils/vector_store.py` | Manages Pinecone index, generates embeddings with `gemini-embedding-001` (768 dims) |
| LLM Chain | `utils/llm_chain.py` | Builds RAG prompts, calls Gemini 2.5 Flash, formats responses with source citations |
| Langfuse Logger | `utils/langfuse_logger.py` | Optional observability â€” logs queries, responses, chunks, and suggested episodes |
| Streamlit UI | `app.py` | Chat interface with sidebar controls and clickable YouTube episode links |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/sammnkwh/lennys-podcast-chatbot.git
cd lennys-podcast-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key (paid tier needed for indexing volume) |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse observability (optional) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse observability (optional) |
| `LANGFUSE_HOST` | No | Langfuse host URL (optional) |

### 4. Index the transcripts

On first run, click **"Index Transcripts"** in the sidebar. This processes ~37K chunks across 303 episodes and stores them in Pinecone. Only needed again if transcripts or chunking strategy change.

### 5. Run the app

```bash
python3 -m streamlit run app.py
```

## Development

```bash
# Run all tests (113 tests, no real API calls needed)
python3 -m pytest tests/ -v

# Run a specific test file
python3 -m pytest tests/test_vector_store.py -v

# Export Langfuse traces to CSV
python3 scripts/export_traces.py --days 7 --output traces_export.csv
```

## Tech Stack

- **UI:** [Streamlit](https://streamlit.io)
- **RAG framework:** [LangChain](https://langchain.com)
- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** `gemini-embedding-001` (768 dimensions)
- **Vector DB:** [Pinecone](https://pinecone.io)
- **Observability:** [Langfuse](https://langfuse.com) (optional)
