# RAG Agent — Build Spec for Claude Max

## Status
Skeleton exists. Core structure, config, chain, and UI are built.
**Missing: the implementation inside each module + .env + wiring it to run.**

## Goal
A working RAG (Retrieval-Augmented Generation) app that:
1. Accepts document uploads (PDF, TXT, MD)
2. Chunks and embeds them into a vector store
3. Answers natural language questions with source citations
4. Runs locally via Streamlit at http://localhost:8501
5. Uses Anthropic Claude as the LLM (we have the key)

This is a **portfolio project** — it needs to look polished and actually work.

---

## What Already Exists (DO NOT rewrite these)

```
src/
  core/
    config.py     ✅ DONE — pydantic settings, all env vars defined
    chain.py      ✅ DONE — RAG chain, prompt, RAGResponse dataclass
  ui/
    app.py        ✅ DONE — full Streamlit UI with chat + upload
  utils/
    logging.py    ✅ EXISTS (check if complete)
```

---

## What Needs to Be Built

### 1. `src/core/embeddings.py`
Build the `build_embeddings(settings)` function.

```python
def build_embeddings(settings: Settings):
    """
    Return an embeddings object based on settings.embedding_provider.
    
    - EmbeddingProvider.HUGGINGFACE → HuggingFaceEmbeddings(model_name=settings.embedding_model)
      Use model "all-MiniLM-L6-v2" (fast, free, no API key needed)
    - EmbeddingProvider.OPENAI → OpenAIEmbeddings(api_key=settings.openai_api_key)
    """
```

Default should be HuggingFace (free, local, no API key).

---

### 2. `src/core/vectorstore.py`
Build these functions:

```python
def build_vectorstore(settings: Settings, embeddings) -> VectorStore:
    """
    Return a vector store instance.
    
    - VectorStoreBackend.CHROMA:
        Load existing from settings.chroma_persist_dir if it exists
        Otherwise create a new empty one
        Use Chroma(persist_directory=..., embedding_function=embeddings)
    
    - VectorStoreBackend.FAISS:
        Try to load from data/vectorstore/faiss_index if it exists
        Otherwise create empty FAISS.from_texts(["placeholder"], embeddings)
    """

def save_faiss_index(vectorstore, settings: Settings):
    """Save FAISS index to disk at data/vectorstore/faiss_index"""
    # Only runs if backend is FAISS
    # Create directory if it doesn't exist
```

---

### 3. `src/core/ingest.py`
Build document ingestion:

```python
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

def ingest_documents(file_paths: list[Path], vectorstore, settings: Settings) -> int:
    """
    Load, chunk, and add documents to the vector store.
    Returns total number of chunks added.
    
    Steps:
    1. Load each file using the right loader:
       - .pdf → PyPDFLoader(str(path))
       - .txt → TextLoader(str(path))
       - .md  → TextLoader(str(path))
    2. Split with RecursiveCharacterTextSplitter(
           chunk_size=settings.chunk_size,
           chunk_overlap=settings.chunk_overlap
       )
    3. Add metadata to each chunk: {"filename": path.name, "source": str(path)}
    4. Add all chunks to vectorstore: vectorstore.add_documents(chunks)
    5. Return len(chunks)
    """
```

---

### 4. `src/core/llm.py`
Build the LLM factory:

```python
def build_llm(settings: Settings) -> BaseChatModel:
    """
    Return the right LLM based on settings.llm_provider.
    
    - LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=0,
        )
    
    - LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0,
            base_url=settings.openai_base_url or None,
        )
    
    - LLMProvider.OLLAMA:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
        )
    
    Raise ValueError if provider is unknown.
    """
```

---

### 5. `.env` file (create at project root)
```env
# LLM Provider: anthropic, openai, or ollama
LLM_PROVIDER=anthropic

# Anthropic
ANTHROPIC_API_KEY=<J_will_fill_this_in>
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# Embeddings (huggingface = free, local)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=data/vectorstore

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_TOP_K=4

# App
APP_HOST=0.0.0.0
APP_PORT=8501
LOG_LEVEL=INFO
```

---

### 6. `src/utils/logging.py` (verify/complete)
Should have:
```python
def setup_logging(level: str = "INFO"):
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
```

---

### 7. Tests — `tests/test_ingest.py` and `tests/test_chain.py`
Write basic smoke tests:
- `test_ingest.py`: test that a sample .txt file gets chunked and added (mock vectorstore)
- `test_chain.py`: test that `format_documents()` works correctly with sample docs

---

## How to Run (after build)

```bash
cd builds/rag-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .env
streamlit run src/ui/app.py
```

Then open http://localhost:8501

---

## README.md — Update With

At the top add:
```
## Quick Start
1. Clone the repo
2. cp .env.example .env and add your ANTHROPIC_API_KEY
3. pip install -r requirements.txt
4. streamlit run src/ui/app.py
5. Upload a PDF and start asking questions
```

Create `.env.example` (same as `.env` but with `ANTHROPIC_API_KEY=your_key_here`).

---

## Definition of Done

- [ ] `streamlit run src/ui/app.py` launches without errors
- [ ] Can upload a PDF and click "Index Documents" — no errors
- [ ] Can type a question and get an answer with sources shown
- [ ] `pytest tests/` passes
- [ ] README has Quick Start section
- [ ] `.env.example` exists

---

## Portfolio Notes (for GitHub README)

This project demonstrates:
- **RAG architecture**: document ingestion → chunking → embedding → retrieval → generation
- **Multi-provider support**: Anthropic Claude, OpenAI, or local Ollama
- **Local-first embeddings**: HuggingFace sentence-transformers (no API key for embeddings)
- **Production patterns**: pydantic-settings config, logging, Dockerized
- **Clean UI**: Streamlit chat interface with source citations
