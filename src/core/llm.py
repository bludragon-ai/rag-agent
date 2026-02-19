"""LLM factory — returns the right chat model based on config."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.core.config import LLMProvider, Settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> BaseChatModel:
    """Return a chat model instance based on settings.llm_provider."""

    if settings.llm_provider == LLMProvider.ANTHROPIC:
        logger.info("Using Anthropic LLM: %s", settings.anthropic_model)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=0,
        )

    if settings.llm_provider == LLMProvider.OPENAI:
        logger.info("Using OpenAI LLM: %s", settings.openai_model)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0,
            base_url=settings.openai_base_url or None,
        )

    if settings.llm_provider == LLMProvider.OLLAMA:
        logger.info("Using Ollama LLM: %s", settings.ollama_model)
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
