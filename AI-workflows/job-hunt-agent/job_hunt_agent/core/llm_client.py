"""Async, provider-agnostic LLM client.

Ported from strategic-reports/strategic_reports/daily/core/llm_client.py —
same litellm + instructor + tenacity pattern, unchanged. See that file for
the full design-decision commentary; this copy keeps only the essential
inline notes to avoid duplicating the same commentary twice in one codebase.
"""

import logging
from typing import TypeVar, Type

import instructor
import litellm
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from pydantic import BaseModel

from job_hunt_agent.core.models import TokenUsage

log = structlog.get_logger(__name__)

litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)

_TENACITY_LOGGER = logging.getLogger(__name__)


class LLMClient:
    """Async, provider-agnostic LLM client built on litellm + instructor.

    Pass any litellm-compatible model string:
      - Ollama:     "ollama_chat/llama3.1:70b"
      - Claude:     "anthropic/claude-sonnet-4-6"
      - OpenAI:     "gpt-4o"
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        run_metadata: dict | None = None,
        instructor_mode: instructor.Mode = instructor.Mode.TOOLS,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._run_metadata = run_metadata or {}
        self._api_base = api_base
        self._api_key = api_key
        self._instructor = instructor.from_litellm(litellm.acompletion, mode=instructor_mode)
        self._total_usage = TokenUsage()

    @property
    def total_usage(self) -> TokenUsage:
        """Read-only view of cumulative token usage across all calls."""
        return self._total_usage

    def _parse_usage(self, completion) -> TokenUsage:
        try:
            u = completion.usage
            return TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
            )
        except Exception:
            return TokenUsage()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(_TENACITY_LOGGER, logging.WARNING),
        reraise=True,
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
    ) -> tuple[T, TokenUsage]:
        """Call the LLM and parse the response into a Pydantic model via instructor."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        log.debug("llm_call_start", model=self.model, schema=response_model.__name__)

        result, completion = await self._instructor.chat.completions.create_with_completion(
            model=self.model,
            response_model=response_model,
            messages=messages,
            temperature=self.temperature,
            **({"api_base": self._api_base} if self._api_base else {}),
            **({"api_key": self._api_key} if self._api_key else {}),
            **({"metadata": self._run_metadata} if self._run_metadata else {}),
        )

        usage = self._parse_usage(completion)
        self._total_usage = self._total_usage + usage

        log.debug(
            "llm_call_complete",
            model=self.model,
            schema=response_model.__name__,
            total_tokens=usage.total_tokens,
        )

        return result, usage
