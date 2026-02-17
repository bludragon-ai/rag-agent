"""
LLM factory — returns the appropriate LangChain chat model
based on the configured provider.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.core.config import Settings, LLMProvider


def build_llm(settings: Settings) -> BaseChatModel:
    """Instantiate the chat model for the active provider."""

    match settings.llm_provider:
        case LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.1,
            )

        case LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                temperature=0.1,
            )

        case LLMProvider.OLLAMA:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.1,
            )

        case _:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
