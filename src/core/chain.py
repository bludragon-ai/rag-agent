"""
RAG chain — the core question-answering pipeline.

Architecture:
  1. User question → vector store retrieval (top-k similar chunks)
  2. Retrieved chunks + question → LLM prompt
  3. LLM generates answer with source citations

The chain is built using LangChain's LCEL (LangChain Expression Language)
for clean composability and streaming support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from src.core.config import Settings

logger = logging.getLogger(__name__)

# ─── Prompt ──────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions based on the provided \
context documents. Follow these rules strictly:

1. Answer ONLY based on the provided context. If the context doesn't contain \
enough information, say so clearly.
2. Cite your sources by referencing the document filename and page (if available) \
in square brackets, e.g. [report.pdf, p.3].
3. Be concise but thorough. Prefer structured answers with bullet points when \
listing multiple items.
4. If different sources contain conflicting information, acknowledge the \
discrepancy and present both perspectives.
5. Never fabricate information not present in the context."""

RAG_USER_TEMPLATE = """\
Context:
{context}

---

Question: {question}

Provide a detailed answer with source citations."""


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[Document]


def format_documents(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        header = f"[Source {i}: {source}"
        if page is not None:
            header += f", p.{page + 1}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(
    llm: BaseChatModel,
    vectorstore: VectorStore,
    settings: Settings,
):
    """
    Construct the RAG chain.

    Returns a callable that takes a question string and returns a RAGResponse.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_TEMPLATE),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    def ask(question: str) -> RAGResponse:
        """Execute the full RAG pipeline for a given question."""
        logger.info("Retrieving documents for: %s", question[:100])

        # Step 1: Retrieve relevant chunks
        docs = retriever.invoke(question)
        logger.info("Retrieved %d chunks", len(docs))

        if not docs:
            return RAGResponse(
                answer="I couldn't find any relevant information in the indexed documents. "
                "Please make sure documents have been uploaded and indexed.",
                sources=[],
            )

        # Step 2: Format context and generate answer
        context = format_documents(docs)
        answer = chain.invoke({"context": context, "question": question})

        return RAGResponse(answer=answer, sources=docs)

    return ask
