"""
LLM provider clients for the pricing agents.

Hackathon standard: all model calls go through the shared endpoint (**Nebius AI Studio** by default; see
the repo-root ``compute.env`` / ``poseidon_compute``). The default model is the thesis model,
``mistralai/Mistral-Large-Instruct-2411``; pass any other model id with ``MODEL_NAME`` / ``--model``.
To run with a model Nebius doesn't serve, use the one entry point ``poseidon-run`` (it hosts it on
Modal or fails loudly). No model is ever substituted.

The thesis's direct Mistral-SDK path (``LLM_PROVIDER=mistral``) and other OpenAI-compatible endpoints are
kept as opt-in alternatives for reproduction or debugging, but participants should not need them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from poseidon_compute import make_client, resolve_model, endpoint_kind, base_url

Message = Dict[str, str]


@dataclass
class Preset:
    base_url: str
    key_env: str
    default_model: str
    needs_key: bool = True
    supports_seed: bool = False


# The thesis used Mistral Large 24.11. Nebius (the default endpoint) serves open-weight models by their
# Hugging Face repo id, so the default here is that HF-cased id — the same model the thesis used, and
# ungated on HF so it can also be Modal-hosted if Nebius doesn't serve it. Pass any other model with
# --model / MODEL_NAME.
THESIS_MODEL = "mistralai/Mistral-Large-Instruct-2411"

PRESETS: Dict[str, Preset] = {
    # Opt-in OpenRouter preset (fallback for closed models / models Nebius doesn't serve).
    # The standard path is the shared compute.env config (Nebius default); see the module docstring.
    "openrouter": Preset(
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        THESIS_MODEL,
        supports_seed=True,
    ),
    # Alternatives (not needed for the hackathon)
    "openai": Preset(
        "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini", supports_seed=True
    ),
    "google": Preset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GOOGLE_API_KEY",
        "gemini-2.5-flash",
    ),
    "mistral-compat": Preset(
        "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-small-latest"
    ),
    "ollama": Preset("http://localhost:11434/v1", "OLLAMA_API_KEY", "llama3.2:3b", needs_key=False),
    # Generic: set LLM_BASE_URL and LLM_API_KEY yourself.
    "openai-compatible": Preset("", "LLM_API_KEY", "", needs_key=False),
}


class FatalProviderError(RuntimeError):
    """Model not found / bad request / bad key: retrying cannot help, abort the run."""


class Provider:
    """Minimal interface every provider implements."""

    name: str = "unknown"
    model_name: str = "unknown"
    #: hosted APIs are rate limited by the Experiment; local servers are not
    is_remote: bool = True

    async def complete(
        self, messages: List[Message], temperature: float, seed: Optional[int]
    ) -> str:
        raise NotImplementedError


class MistralProvider(Provider):
    """Original thesis path: the official Mistral SDK."""

    name = "mistral"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set (LLM_PROVIDER=mistral)")
        self.model_name = model or os.getenv("MODEL_NAME") or "mistral-large-2411"

    async def complete(self, messages, temperature, seed):
        from mistralai import Mistral

        async with Mistral(api_key=self.api_key) as client:
            kwargs = {}
            if seed is not None:
                kwargs["random_seed"] = seed
            response = await client.chat.complete_async(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs,
            )
        return response.choices[0].message.content


class OpenAICompatibleProvider(Provider):
    """OpenAI, Gemini, OpenRouter, Mistral's REST API, Ollama, LM Studio, vLLM..."""

    def __init__(
        self,
        preset: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if preset not in PRESETS:
            raise ValueError(f"Unknown provider {preset!r}. Choose from {sorted(PRESETS)}")
        p = PRESETS[preset]
        self.name = preset
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or p.base_url
        if not self.base_url:
            raise ValueError("LLM_BASE_URL must be set for provider 'openai-compatible'")
        # The client authenticates with the key from the shared compute.env (via make_client() /
        # llm_config, Nebius or OpenRouter), which fails loudly if none is set. This field is only a
        # diagnostic hint for the preset; it is not what authenticates the client.
        self.api_key = api_key or os.getenv(p.key_env)  # diagnostics only; real key comes from llm_config
        # Route the explicit --model / MODEL_NAME through resolve_model too (not just the preset
        # default), so the LLM_MODEL override, retired-pin upgrades and OpenRouter id-casing apply to it.
        self.model_name = resolve_model(model or os.getenv("MODEL_NAME") or p.default_model)
        if not self.model_name:
            raise ValueError("MODEL_NAME must be set for provider 'openai-compatible'")
        self.supports_seed = p.supports_seed
        self.is_remote = "localhost" not in self.base_url and "127.0.0.1" not in self.base_url

        # The default 'openrouter' preset routes through the shared compute.env config (Nebius by
        # default). The opt-in presets (openai / google / mistral-compat / ollama / openai-compatible)
        # name a DIFFERENT endpoint, so honor their resolved base_url/key instead of silently rerouting
        # everything to the shared endpoint (which also makes is_remote above load-bearing again).
        # Optional OpenRouter provider pin, e.g. OPENROUTER_PROVIDER=Mistral (comma-separated for an
        # ordered list). Fallbacks are switched off so a run never silently lands on another host.
        self.routing = None
        if preset == "openrouter":
            self._client = make_client(async_=True)
            pinned = [p.strip() for p in os.getenv("OPENROUTER_PROVIDER", "").split(",") if p.strip()]
            if pinned and endpoint_kind() == "openrouter":
                self.routing = {"order": pinned, "allow_fallbacks": False}
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key or "EMPTY")
        self._json_mode = True  # drop response_format if the model rejects it

    async def complete(self, messages, temperature, seed):
        kwargs = {}
        if seed is not None and self.supports_seed:
            kwargs["seed"] = seed
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.routing:
            kwargs["extra_body"] = {"provider": self.routing}
        from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

        fatal = (NotFoundError, AuthenticationError, PermissionDeniedError)
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=temperature, **kwargs
            )
        except fatal as e:
            raise FatalProviderError(
                f"{self.model_name}: {e}. Check the model id and your NEBIUS_API_KEY in the repo-root "
                f"compute.env. If the model isn't on Nebius, run via `poseidon-run --model "
                f"{self.model_name} -- <cmd>` to host it on Modal (or fail with a clear reason)."
            ) from e
        except BadRequestError as e:
            if self._json_mode and "response_format" in str(e).lower():
                self._json_mode = False  # model does not support JSON mode; rely on the prompt
                kwargs.pop("response_format")
                response = await self._client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=temperature, **kwargs
                )
            else:
                raise FatalProviderError(f"{self.model_name}: {e}") from e
        if not response.choices:
            raise RuntimeError(f"{self.model_name}: empty response")
        return strip_code_fence(response.choices[0].message.content or "")


def strip_code_fence(text: str) -> str:
    """Remove <think>...</think> blocks (reasoning models) and ```json fences around the answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def make_provider(
    kind: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None
) -> Provider:
    """Build a provider from ``kind`` (or the ``LLM_PROVIDER`` env var, default openrouter)."""
    kind = (kind or os.getenv("LLM_PROVIDER", "openrouter")).lower()
    if kind == "mistral":
        return MistralProvider(api_key=api_key, model=model)
    return OpenAICompatibleProvider(preset=kind, model=model, api_key=api_key)


def resolve_effective_model(model: Optional[str] = None) -> str:
    """The model id a run will actually use, via the same path as ``make_provider``.

    Mirrors ``OpenAICompatibleProvider.__init__``: an explicit ``--model`` / ``MODEL_NAME``
    wins over the thesis default, and whichever id is chosen is passed through ``resolve_model``
    so ``LLM_MODEL`` overrides, retired-pin upgrades and OpenRouter id-translation are reflected.
    This keeps the pre-run cost banner honest about the resolved model (no silent swap).
    """
    return resolve_model(model or os.getenv("MODEL_NAME") or THESIS_MODEL)


def estimate_cost_note(n_calls: int, model: Optional[str] = None) -> str:
    model = resolve_effective_model(model)
    # Show the rates page for the endpoint the call will ACTUALLY use (Nebius, or OpenRouter when
    # that's the configured key), not a hardcoded provider — otherwise the banner misleads OpenRouter users.
    kind = endpoint_kind()
    if kind == "openrouter":
        rates = f"https://openrouter.ai/{model}"
    elif kind == "nebius":
        rates = "Nebius AI Studio (https://studio.nebius.com)"
    else:  # custom endpoint (e.g. a Modal-hosted model)
        rates = base_url()
    return f"≈ {n_calls:,} model calls with {model}. Per-token rates: {rates}."
