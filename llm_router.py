"""
Multi-provider LLM router using LiteLLM.
Supports host-provided API keys and optional BYOK (user-supplied keys).
"""
import os
from typing import Any, Dict, Generator, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Model registry — ids match LiteLLM model strings
MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "groq/llama-3.1-8b-instant",
        "name": "Groq Llama 3.1 8B (Fast)",
        "provider": "groq",
        "env_key": "GROQ_API_KEY",
        "description": "Fast inference, good for general queries",
    },
    {
        "id": "groq/llama-3.3-70b-versatile",
        "name": "Groq Llama 3.3 70B",
        "provider": "groq",
        "env_key": "GROQ_API_KEY",
        "description": "Higher quality, still fast on Groq",
    },
    {
        "id": "gpt-4o-mini",
        "name": "OpenAI GPT-4o Mini",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "description": "Cost-efficient OpenAI model",
    },
    {
        "id": "gpt-4o",
        "name": "OpenAI GPT-4o",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "description": "Premium accuracy for complex queries",
    },
    {
        "id": "gemini/gemini-2.0-flash",
        "name": "Google Gemini 2.0 Flash",
        "provider": "gemini",
        "env_key": "GEMINI_API_KEY",
        "description": "Long-context, fast Google model",
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek Chat",
        "provider": "deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "description": "Cost-efficient alternative",
    },
]

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"

_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _get_model_config(model_id: str) -> Optional[Dict[str, Any]]:
    for m in MODEL_REGISTRY:
        if m["id"] == model_id:
            return m
    return None


def _resolve_api_key(
    model_id: str,
    user_api_key: Optional[str] = None,
) -> tuple[str, bool]:
    """Return (api_key, is_byok). Raises ValueError if no key available."""
    if user_api_key and user_api_key.strip():
        return user_api_key.strip(), True

    config = _get_model_config(model_id)
    if not config:
        raise ValueError(f"Unknown model: {model_id}")

    host_key = os.getenv(config["env_key"])
    if host_key:
        return host_key, False

    raise ValueError(
        f"No API key available for {config['name']}. "
        "Provide your own key (BYOK) or ask the host to configure server keys."
    )


def get_available_models() -> List[Dict[str, Any]]:
    """Return models list with host_availability flag."""
    result = []
    for m in MODEL_REGISTRY:
        host_available = bool(os.getenv(m["env_key"]))
        result.append({
            "id": m["id"],
            "name": m["name"],
            "provider": m["provider"],
            "description": m["description"],
            "host_available": host_available,
            "byok_supported": True,
        })
    return result


def chat_completion(
    messages: List[Dict[str, str]],
    model_id: str = DEFAULT_MODEL,
    user_api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = False,
) -> Any:
    """Unified chat completion via LiteLLM."""
    import litellm

    api_key, is_byok = _resolve_api_key(model_id, user_api_key)
    config = _get_model_config(model_id) or {"provider": "groq"}

    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # Pass provider-specific key to LiteLLM
    provider = config["provider"]
    if is_byok or not os.getenv(config.get("env_key", "")):
        env_name = _PROVIDER_KEY_ENV.get(provider, config.get("env_key", ""))
        if env_name:
            kwargs["api_key"] = api_key

    return litellm.completion(**kwargs)


def chat_completion_stream(
    messages: List[Dict[str, str]],
    model_id: str = DEFAULT_MODEL,
    user_api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """Stream tokens from LiteLLM completion."""
    response = chat_completion(
        messages=messages,
        model_id=model_id,
        user_api_key=user_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def simple_completion(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    user_api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Single-turn completion (used for query reformulation)."""
    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model_id=model_id,
        user_api_key=user_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content.strip()
