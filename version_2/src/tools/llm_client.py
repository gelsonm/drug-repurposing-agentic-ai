"""
LLM Client Factory for Drug Repurposing v2.

Tries OpenAI (GPT-4o-mini) first.
If the OpenAI key is missing or the call fails, falls back to Groq
(llama-3.3-70b-versatile — fast, free tier available).

Usage:
    from src.tools.llm_client import get_llm_client, llm_chat

    client, provider = get_llm_client()
    response = llm_chat(client, provider, messages=[...], max_tokens=800)
"""
from __future__ import annotations

from typing import Any, Literal

from loguru import logger

Provider = Literal["openai", "groq"]


def get_llm_client() -> tuple[Any, Provider]:
    """
    Returns (client, provider) for the best available LLM.

    Priority:
      1. OpenAI — if OPENAI_API_KEY is set and non-empty
      2. Groq   — if GROQ_API_KEY is set and non-empty
      3. Raises RuntimeError if neither is available

    Returns:
        (client, provider_name)  where provider_name is "openai" or "groq"
    """
    from src.config import get_settings
    settings = get_settings()

    # ── Try OpenAI first ──────────────────────────────────────────────────
    if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            # Quick connectivity check (no API call, just client init)
            logger.info(f"LLM provider: OpenAI ({settings.openai_model})")
            return client, "openai"
        except Exception as e:
            logger.warning(f"OpenAI client init failed: {e} — trying Groq fallback")

    # ── Try Groq fallback ─────────────────────────────────────────────────
    if settings.groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            logger.info(f"LLM provider: Groq ({settings.groq_model}) [fallback]")
            return client, "groq"
        except ImportError:
            logger.error(
                "Groq package not installed. Run: uv pip install groq --python <venv>/python.exe"
            )
        except Exception as e:
            logger.warning(f"Groq client init failed: {e}")

    raise RuntimeError(
        "No LLM provider available. Set OPENAI_API_KEY or GROQ_API_KEY in .env"
    )


def llm_chat(
    client: Any,
    provider: Provider,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 800,
) -> str:
    """
    Unified chat completion call that works for both OpenAI and Groq clients.
    Both expose an identical `client.chat.completions.create()` API.

    Args:
        client: OpenAI or Groq client instance
        provider: "openai" or "groq"
        messages: List of {role, content} dicts
        model: Override model name (uses settings default if None)
        temperature: Override temperature (uses settings default if None)
        max_tokens: Max tokens for completion

    Returns:
        Response text string, or empty string on failure.
    """
    from src.config import get_settings
    settings = get_settings()

    if model is None:
        model = settings.openai_model if provider == "openai" else settings.groq_model
    if temperature is None:
        temperature = settings.openai_temperature

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        if provider == "openai":
            logger.warning(f"OpenAI chat failed ({type(e).__name__}: {e})")
            # Attempt Groq fallback mid-conversation
            settings = get_settings()
            if settings.groq_api_key:
                logger.info("Attempting Groq fallback after OpenAI failure...")
                try:
                    from groq import Groq
                    groq_client = Groq(api_key=settings.groq_api_key)
                    groq_response = groq_client.chat.completions.create(
                        model=settings.groq_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    logger.info("Groq fallback succeeded.")
                    return groq_response.choices[0].message.content or ""
                except Exception as e2:
                    logger.warning(f"Groq fallback also failed: {e2}")
        else:
            logger.warning(f"Groq chat failed ({type(e).__name__}: {e})")

        return ""


def get_provider_display_name(provider: Provider) -> str:
    """Human-readable provider name for UI display."""
    from src.config import get_settings
    settings = get_settings()
    if provider == "openai":
        return f"OpenAI {settings.openai_model}"
    return f"Groq {settings.groq_model}"
