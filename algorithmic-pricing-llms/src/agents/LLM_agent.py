"""
LLM Agent Implementation for Market Simulation

This module provides the LLMAgent class that turns a market prompt into a
pricing decision through an LLM provider (see ``llm_providers.py``).

Hackathon note: the thesis used the Mistral SDK directly; ``provider`` now
selects Mistral (default), OpenAI, Google, OpenRouter, or any OpenAI-
compatible endpoint (including a free local Ollama server). ``temperature``
and ``seed`` are exposed for the one-variable-at-a-time ablations.
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Type

from pydantic import BaseModel, ValidationError, create_model

from src.agents.base_agent import Agent
from src.agents.llm_providers import FatalProviderError, Provider, make_provider

MAX_RETRIES = 15
RETRY_DELAY_SECONDS = 1


class LLMAgent(Agent):
    """
    LLM-based agent for strategic pricing decisions in oligopoly markets.

    Args:
        name: Unique identifier for the agent
        prefix: Prompt prefix defining agent's strategic context (P1 / P2)
        response_model: Pydantic model for response validation
        provider: an ``llm_providers.Provider``; default ``make_provider()``
            which reads ``LLM_PROVIDER`` / ``MODEL_NAME`` from the env
        api_key / model_name: kept for compatibility with the thesis scripts;
            forwarded to ``make_provider`` when ``provider`` is None
        prompt_template: Template for prompt formatting
        memory_length: Number of past rounds shown in the prompt (0 = none)
        history_order: order of those rounds in the prompt, "newest" (thesis) or "oldest" first
        temperature: sampling temperature (thesis: 0.7)
        seed: sampling seed, forwarded when the provider supports it
        env_params: Market environment parameters
        logger: Logger instance for experiment tracking
    """

    def __init__(
        self,
        name: str,
        prefix: str,
        response_model: Type[BaseModel],
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_template=None,
        memory_length: int = 100,
        history_order: str = "newest",
        temperature: float = 0.7,
        seed: Optional[int] = None,
        env_params: dict = None,
        logger: logging.Logger = None,
    ):
        super().__init__(
            name=name,
            prefix=prefix,
            prompt_template=prompt_template,
            env_params=env_params,
            logger=logger,
        )
        self.provider = provider or make_provider(model=model_name, api_key=api_key)
        self.model_name = self.provider.model_name
        self.temperature = temperature
        self.seed = seed
        self.response_model = response_model
        self.reponse_model_in_answer = self.__filter_in_answer_fields(response_model)
        self.prompt_template = prompt_template
        self.memory_length = memory_length
        if history_order not in ("newest", "oldest"):
            raise ValueError(f"history_order must be 'newest' or 'oldest', got {history_order!r}")
        self.history_order = history_order
        self.logger = logger or logging.getLogger("experiment_logger")
        self.reponse_model_str = (
            "Respond only with a JSON object with this schema:\n{\n"
            + "\n".join(
                [
                    f'  "{name}": {field.annotation.__name__}'
                    for name, field in response_model.model_fields.items()
                    if field.json_schema_extra.get("in_answer", True)
                ]
            )
            + "\n}"
        )

    @property
    def is_remote(self) -> bool:
        return getattr(self.provider, "is_remote", True)

    def build_messages(self, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": self.reponse_model_str},
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": prompt},
        ]

    async def act(self, prompt: str) -> Dict:
        """Call the provider with retries and validate the JSON answer."""
        messages = self.build_messages(prompt)
        for attempt in range(1, MAX_RETRIES + 1):
            self.logger.info(f"🔄 Attempt {attempt} for agent {self.name}")
            try:
                raw = await self.provider.complete(messages, self.temperature, self.seed)
                parsed_json = json.loads(raw)
                validated = self.reponse_model_in_answer(**parsed_json)
                return {"agent_name": self.name, "content": validated.model_dump()}
            except FatalProviderError:
                raise  # model id / key problems: retrying cannot help
            except (ValidationError, ValueError) as e:
                if attempt == MAX_RETRIES:
                    self.logger.error(
                        f"❌ Validation failed after {MAX_RETRIES} attempts for agent {self.name}: {e}"
                    )
                    raise e
            except Exception as e:
                if attempt == MAX_RETRIES:
                    self.logger.error(
                        f"❌ Error after {MAX_RETRIES} attempts for agent {self.name}: {e}"
                    )
                    raise e
            await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

    def __filter_in_answer_fields(self, model: Type[BaseModel]) -> Type[BaseModel]:
        in_answer_fields = {
            name: (field.annotation, field.default)
            for name, field in model.model_fields.items()
            if field.json_schema_extra.get("in_answer", False)
        }
        return create_model(model.__name__ + "Filtered", **in_answer_fields)

    @property
    def type(self) -> str:
        return "LLM_agent"
