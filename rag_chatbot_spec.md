# RAG Chatbot with Follow-up Questions and Eval Framework

## Project Overview
Build a customizable RAG (Retrieval-Augmented Generation) chatbot using LangChain and Streamlit with intelligent follow-up question generation and built-in evaluation capabilities.

## Core Requirements

### Technology Stack
- **Framework:** LangChain (Python)
- **UI:** Streamlit
- **LLM:** Google Gemini (free tier)
  - Model: Gemini 1.5 Flash or Gemini 2.0 Flash
  - Embeddings: text-embedding-004
- **Vector Database:** Pinecone (cloud, free tier)
  - Alternative option: ChromaDB (local) - make it configurable
- **Deployment:** Streamlit Community Cloud (free)

### Key Features

#### 1. Document Management
- Upload documents (PDF, TXT, MD formats)
- Automatic chunking and embedding
- Store in Pinecone vector database
- Display uploaded documents list

#### 2. Customizable System Prompt
- Editable text area in UI for system prompt
- Default prompt should instruct the LLM to:
  - Answer questions based on retrieved context
  - Generate 3-5 relevant follow-up questions after each answer
  - Make follow-ups specific to the document content AND conversation context
- Save/load different prompt templates

#### 3. Multi-turn Conversation
- Full conversation memory using LangChain's ConversationBufferMemory
- Chat interface with message history
- Context carries across turns
- "New Chat" button to reset conversation

#### 4. Follow-up Question Buttons
- After each bot response, display 3-5 clickable follow-up question buttons
- Clicking a button automatically submits that question
- Follow-ups should be contextually relevant
- Clean, intuitive UI (similar to ChatGPT suggested prompts)

#### 5. Evaluation Framework
- Log every interaction with:
  - Timestamp
  - User question
  - Retrieved context chunks
  - Bot answer
  - Generated follow-up questions
  - Whether user clicked a follow-up (engagement tracking)
  - Manual quality rating (1-5 stars)
- Separate "Eval Dashboard" tab in Streamlit showing:
  - All logged interactions
  - Ability to rate follow-up quality
  - Basic metrics (avg rating, engagement rate, response time)
  - Export to CSV/JSON
- Store eval data in SQLite database or JSON files

#### 6. Optional Langfuse Integration
- Include code (commented out by default) for Langfuse integration
- Simple toggle to enable/disable tracking
- Instructions for setting up Langfuse in comments

## Technical Architecture

### File Structure
```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── README.md             # Setup and usage instructions
├── utils/
│   ├── __init__.py
│   ├── document_processor.py   # Document loading and chunking
│   ├── vector_store.py         # Pinecone/ChromaDB operations
│   ├── llm_chain.py           # LangChain RAG chain setup
│   └── eval_logger.py         # Logging and eval data management
└── data/
    ├── uploads/          # Temporary uploaded files
    └── eval_data/        # SQLite DB or JSON files
```

### Code Components

#### Document Processing
- Use LangChain's document loaders (PyPDFLoader, TextLoader, etc.)
- Chunk documents using RecursiveCharacterTextSplitter
- Chunk size: 1000 characters, overlap: 200 characters
- Generate embeddings using Google's text-embedding-004

#### Vector Store Setup
- Pinecone index creation (if doesn't exist)
- Store document chunks with metadata (source, page, chunk_id)
- Efficient similarity search (retrieve top 4-5 chunks)
- Support switching between Pinecone (cloud) and ChromaDB (local)

#### LLM Chain Architecture
- Use LangChain's ConversationalRetrievalChain
- Combine retrieved context + conversation history + custom prompt
- Prompt template structure:
  ```
  System: {custom_system_prompt}
  
  Context from documents: {context}
  
  Conversation history: {chat_history}
  
  User question: {question}
  
  Instructions: 
  1. Answer the question based on the context
  2. Generate 3-5 relevant follow-up questions
  3. Format follow-ups as a numbered list at the end
  ```
- Parse LLM response to extract answer and follow-up questions

#### Streamlit UI Layout
**Sidebar:**
- Document upload widget
- List of uploaded documents
- Vector database selection (Pinecone/ChromaDB)
- "New Chat" button
- Settings (temperature, max tokens, etc.)

**Main Area - Chat Tab:**
- System prompt editor (expandable text area)
- Chat message history (user and assistant messages)
- Follow-up question buttons (displayed after each assistant message)
- Chat input box at bottom

**Main Area - Eval Dashboard Tab:**
- Table of logged interactions
- Rating interface for each interaction
- Basic metrics visualization
- Export button (CSV/JSON)

#### Evaluation System
- SQLite schema:
  ```sql
  CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    user_question TEXT,
    retrieved_chunks TEXT,
    bot_answer TEXT,
    followup_questions TEXT,
    followup_clicked BOOLEAN,
    clicked_question TEXT,
    quality_rating INTEGER,
    response_time_ms INTEGER
  )
  ```
- Auto-log every interaction
- Manual rating interface in dashboard
- Calculate metrics: avg rating, engagement rate, response time

## Environment Variables
```
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
# Optional
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Dependencies (requirements.txt)
```
streamlit>=1.30.0
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.20
pinecone-client>=3.0.0
chromadb>=0.4.22
pypdf>=4.0.0
python-dotenv>=1.0.0
# Optional
langfuse>=2.0.0
```

## Success Criteria
- [ ] Can upload and process PDF/TXT documents
- [ ] Documents are successfully embedded and stored in vector database
- [ ] Custom system prompt is editable and affects responses
- [ ] Chat maintains conversation context across turns
- [ ] Follow-up questions are generated after each response
- [ ] Follow-up buttons are clickable and submit questions
- [ ] All interactions are logged with metadata
- [ ] Eval dashboard displays logged data
- [ ] Can manually rate interaction quality
- [ ] Can export eval data to CSV
- [ ] Langfuse integration works when enabled
- [ ] Entire setup takes <15 minutes after dependencies installed
- [ ] Code is well-commented and modular

## Timeline Goal
- Development: ~15-20 minutes
- Testing and iteration: ~30-40 minutes
- Total: Under 1 hour to fully working chatbot

## Additional Notes
- Prioritize clean, readable code over optimization
- Include helpful comments explaining each component
- Make it easy to swap components (LLM, vector DB, etc.)
- Design for extensibility (easy to add new features later)
- Include error handling for common issues (API limits, upload failures, etc.)
- Provide clear setup instructions in README

## Future Enhancement Ideas (Not Required Now)
- A/B testing different prompt templates
- Automatic eval metrics (relevance scoring using another LLM call)
- Support for more document types (DOCX, HTML, etc.)
- Chat history persistence across sessions
- Multi-document comparison queries
- Integration with Google Drive for document import
- Deployment instructions for Streamlit Cloud

## Design Philosophy
This chatbot serves dual purposes:
1. **Functional tool:** Actually useful for querying documents
2. **Research platform:** Test and evaluate follow-up question generation quality

The eval framework is as important as the chatbot itself - it should make it easy to iterate on prompt engineering and measure improvement in follow-up question quality.
