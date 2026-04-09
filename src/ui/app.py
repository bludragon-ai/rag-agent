"""
Streamlit UI for RAG Agent.

A clean, professional interface for document upload and question answering
with source citations.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.core.config import get_settings
from src.core.embeddings import build_embeddings
from src.core.ingest import ingest_documents, SUPPORTED_EXTENSIONS
from src.core.llm import build_llm
from src.core.chain import build_rag_chain, RAGResponse
from src.core.vectorstore import build_vectorstore, save_faiss_index
from src.utils.logging import setup_logging

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Portfolio Theme ────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { background-color: #000000; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace;
        color: #d4a853;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 500;
    }

    /* Main title */
    .stApp h1 {
        color: white;
        font-weight: 700;
    }

    /* Subtitle/description text */
    .stApp .stMarkdown p {
        color: #a3a3a3;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #111111 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #e5e5e5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(212,168,83,0.3) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: rgba(212,168,83,0.1);
        color: #d4a853;
        border: 1px solid rgba(212,168,83,0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: rgba(212,168,83,0.2);
        border-color: rgba(212,168,83,0.5);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }

    /* Tables */
    .stTable, .stDataFrame {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        overflow: hidden;
    }
    .stTable th {
        background-color: #111111;
        color: #d4a853;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stTable td {
        background-color: #0a0a0a;
        color: #a3a3a3;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.04);
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #000000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }

    /* Metric styling */
    [data-testid="stMetric"] label {
        font-family: 'JetBrains Mono', monospace;
        color: #d4a853;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialization ─────────────────────────────────────────────────────────


@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline (cached across reruns)."""
    setup_logging()
    settings = get_settings()
    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings, embeddings)
    llm = build_llm(settings)
    ask = build_rag_chain(llm, vectorstore, settings)
    return settings, vectorstore, ask


# ─── Sidebar: Document Upload ───────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Document Manager")
    st.caption("Upload documents to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown",
    )

    if uploaded_files and st.button("📥 Index Documents", type="primary", use_container_width=True):
        settings, vectorstore, ask = init_pipeline()

        with st.spinner("Processing documents..."):
            # Save uploaded files to temp directory
            temp_dir = Path(tempfile.mkdtemp())
            file_paths = []
            for f in uploaded_files:
                path = temp_dir / f.name
                path.write_bytes(f.read())
                file_paths.append(path)

            # Ingest
            num_chunks = ingest_documents(file_paths, vectorstore, settings)
            save_faiss_index(vectorstore, settings)

            # Clean up temp files
            shutil.rmtree(temp_dir)

        st.success(f"✅ Indexed **{num_chunks}** chunks from **{len(file_paths)}** file(s).")

    st.divider()

    # Status
    st.header("⚙️ Configuration")
    try:
        settings = get_settings()
        st.markdown(f"""
        | Setting | Value |
        |---------|-------|
        | **LLM** | `{settings.llm_provider.value}` |
        | **Model** | `{getattr(settings, f'{settings.llm_provider.value}_model', 'n/a')}` |
        | **Vector Store** | `{settings.vector_store.value}` |
        | **Chunk Size** | `{settings.chunk_size}` |
        | **Top-K** | `{settings.retrieval_top_k}` |
        """)
    except Exception:
        st.warning("Configure your `.env` file to get started.")

# ─── Main: Chat Interface ───────────────────────────────────────────────────

st.markdown('<p style="font-family: JetBrains Mono, monospace; color: #d4a853; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 0;">Interactive Demo</p>', unsafe_allow_html=True)
st.title("🔍 RAG Agent")
st.caption("Ask questions about your documents. Answers include source citations.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    name = src.get("filename", src.get("source", "unknown"))
                    page = src.get("page")
                    label = f"**{name}**" + (f" (p.{page + 1})" if page is not None else "")
                    st.markdown(f"{i}. {label}")
                    st.code(src["content"][:300] + ("..." if len(src["content"]) > 300 else ""), language=None)

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        try:
            settings, vectorstore, ask = init_pipeline()

            with st.spinner("Searching documents and generating answer..."):
                response: RAGResponse = ask(question)

            st.markdown(response.answer)

            # Source details
            source_data = []
            if response.sources:
                with st.expander("📚 Sources", expanded=False):
                    for i, doc in enumerate(response.sources, 1):
                        name = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
                        page = doc.metadata.get("page")
                        label = f"**{name}**" + (f" (p.{page + 1})" if page is not None else "")
                        st.markdown(f"{i}. {label}")
                        st.code(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""), language=None)
                        source_data.append({
                            "filename": name,
                            "page": page,
                            "source": doc.metadata.get("source", ""),
                            "content": doc.page_content,
                        })

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": source_data,
            })

        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
