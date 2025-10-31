from __future__ import annotations

import os
from abc import ABC, abstractmethod
import re
from typing import Any, Callable, Dict, Optional

try:
    from src.utils.prompts import get_game_prompt
except Exception:  # pragma: no cover - optional import during setup
    get_game_prompt = None  # type: ignore


STANDARD_GAME_PROMPT = """You are a competitive game player. Follow these strict instructions:
1. Read the game rules and current state carefully.
2. ONLY output your next move in the required format.
3. Never add commentary, explanations, or predictions."""


_GAME_SIGNATURES: Dict[str, tuple[list[str], list[str]]] = {
    "secret_mafia": (
        ["secret mafia", "welcome to secret mafia", "night phase"],
        ["mafia", "villagers"]
    ),
    "colonel_blotto": (
        ["colonel blotto", "colonelblotto", "commander alpha", "=== colonel blotto"],
        ["units to allocate", "format: '[a4 b2 c2]'"]
    ),
    "three_player_ipd": (
        ["iterated prisoner's dilemma", "three-player ipd", "3-player iterated"],
        ["payoff matrix", "cooperate"]
    ),
    "codenames": (
        ["codenames", "you are playing codenames"],
        ["spymaster", "operative"]
    ),
}


def _detect_game_from_observation(observation: str) -> Optional[str]:
    """Best-effort game detection from raw arena observations."""
    if not observation:
        return None

    lower_obs = observation.lower()
    game_lines = [match.strip() for match in re.findall(r"\[game\](.*)", lower_obs)]
    search_spaces = [lower_obs, *game_lines]

    def _contains(tokens: list[str]) -> bool:
        return any(token in space for space in search_spaces for token in tokens)

    for game, (primary_tokens, secondary_tokens) in _GAME_SIGNATURES.items():
        if _contains(primary_tokens):
            return game
        if secondary_tokens and _contains(secondary_tokens):
            return game

    return None


class Agent(ABC):
    """Minimal interface for all agents."""

    @abstractmethod
    def __call__(self, observation: str) -> str:
        """Produce the next action for the given observation."""


class OpenAIAgent(Agent):
    """Thin wrapper around the OpenAI Chat Completions API with automatic prompt routing."""

    DEFAULT_TIMEOUT = 300.0
    DEFAULT_MAX_TOKENS = 10000
    DEFAULT_CONTEXT_WINDOW = 16000
    GENERATION_MARGIN = 100

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        request_timeout: Optional[float] = None,
        prompt_variant: Optional[str] = None,
        prompt_loader: Optional[Callable[[str, Optional[str]], Optional[str]]] = None,
        enable_thinking: bool = True,
        **completion_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Chat-completions capable model identifier.
            api_key: API key (falls back to OPENAI_API_KEY env).
            base_url: Optional custom endpoint (falls back to OPENAI_BASE_URL env).
            system_prompt: Prompt injected as the system message when routing is disabled.
            request_timeout: Optional override for HTTP timeout.
            prompt_variant: Optional variant name (e.g., "strong") passed to the prompt loader.
            prompt_loader: Custom loader to resolve prompts; defaults to `get_game_prompt`.
                When the loader is available and ``system_prompt`` is not provided, the
                agent will attempt to infer the game from each observation and fetch the
                corresponding prompt (falling back to a generic baseline if unresolved).
            **completion_kwargs: Forwarded to ``client.chat.completions.create``.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("OpenAI package is required. Install it with: pip install openai") from exc

        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._static_system_prompt = system_prompt
        self._fallback_system_prompt = STANDARD_GAME_PROMPT
        completion_params = dict(completion_kwargs)
        self._context_window = completion_params.pop(
            "context_window", self.DEFAULT_CONTEXT_WINDOW
        )
        self._generation_margin = completion_params.pop(
            "context_margin", self.GENERATION_MARGIN
        )
        self._tokenizer_name = completion_params.pop("tokenizer_name", self.model_name)
        self._completion_kwargs = self._inject_thinking_flag(
            completion_params, enable_thinking
        )
        self._completion_kwargs.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)

        try:
            self._context_window = int(self._context_window)
        except (TypeError, ValueError):
            self._context_window = self.DEFAULT_CONTEXT_WINDOW
        try:
            self._generation_margin = max(0, int(self._generation_margin))
        except (TypeError, ValueError):
            self._generation_margin = self.GENERATION_MARGIN

        self.prompt_variant = prompt_variant
        self._prompt_loader: Optional[
            Callable[[str, Optional[str]], Optional[str]]
        ] = prompt_loader or get_game_prompt
        self._tokenizer_initialized = False
        self._tokenizer: Any = None

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            resolved_key = "sk-placeholder"

        resolved_base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        timeout = request_timeout if request_timeout is not None else self.DEFAULT_TIMEOUT

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base, timeout=timeout)

    @staticmethod
    def _inject_thinking_flag(
        completion_kwargs: Dict[str, Any], enable_thinking: bool
    ) -> Dict[str, Any]:
        if not enable_thinking:
            return completion_kwargs
        extra_body = dict(completion_kwargs.get("extra_body") or {})
        chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("enable_thinking", True)
        extra_body["chat_template_kwargs"] = chat_kwargs
        completion_kwargs["extra_body"] = extra_body
        return completion_kwargs

    def _resolve_system_prompt(self, observation: str) -> str:
        if self._static_system_prompt is not None:
            return self._static_system_prompt

        if self._prompt_loader and observation:
            game = _detect_game_from_observation(observation)
            if game:
                prompt = self._prompt_loader(game, self.prompt_variant)
                if prompt:
                    if game == "colonel_blotto":
                        prompt = prompt.replace("Format: '[A4 B2 C2]'", "Format: '[A7 B7 C6]'")
                    return prompt

        return self._fallback_system_prompt

    def _normalize_observation(self, observation: str) -> str:
        if not observation:
            return observation
        game = _detect_game_from_observation(observation)
        if game == "colonel_blotto":
            return observation.replace("Format: '[A4 B2 C2]'", "Format: '[A7 B7 C6]'")
        return observation

    def __call__(self, observation: str) -> str:
        normalized_observation = self._normalize_observation(observation)
        messages = [
            {"role": "system", "content": self._resolve_system_prompt(normalized_observation)},
            {"role": "user", "content": normalized_observation},
        ]
        self._ensure_tokenizer_state()
        request_kwargs = dict(self._completion_kwargs)
        configured_max = request_kwargs.get("max_tokens")
        request_kwargs["max_tokens"] = self._adjust_max_tokens(messages, configured_max)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **request_kwargs,
            )
        except Exception:
            return "[ERROR]"

        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
            if cleaned:
                return cleaned
        return "[ERROR]"

    def _adjust_max_tokens(
        self,
        messages: list[Dict[str, str]],
        configured_max: Optional[int],
    ) -> int:
        fallback = configured_max or self.DEFAULT_MAX_TOKENS
        prompt_tokens = self._count_prompt_tokens(messages)
        if prompt_tokens is None:
            return fallback

        available = self._context_window - prompt_tokens - self._generation_margin
        if available <= 0:
            return 1
        return max(1, min(fallback, available))

    def _count_prompt_tokens(self, messages: list[Dict[str, str]]) -> Optional[int]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            encoded = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
        except Exception:
            return None

        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = getattr(encoded, "input_ids", None)
            if input_ids is None and hasattr(encoded, "__getitem__"):
                try:
                    input_ids = encoded["input_ids"]
                except KeyError:
                    input_ids = None
        if input_ids is None:
            return None

        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            return len(input_ids)

        if hasattr(input_ids, "shape"):
            shape = input_ids.shape
            if shape:
                return int(shape[-1])

        try:
            return len(input_ids)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _ensure_tokenizer_state(self) -> None:
        if not hasattr(self, "_context_window") or self._context_window is None:
            self._context_window = self.DEFAULT_CONTEXT_WINDOW
        if not hasattr(self, "_generation_margin") or self._generation_margin is None:
            self._generation_margin = self.GENERATION_MARGIN
        if not hasattr(self, "_tokenizer_name") or not self._tokenizer_name:
            self._tokenizer_name = self.model_name
        if not hasattr(self, "_tokenizer_initialized"):
            self._tokenizer_initialized = False
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer_initialized:
            return self._tokenizer

        self._tokenizer_initialized = True
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            self._tokenizer = None
            return None

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer


__all__ = [
    "Agent",
    "OpenAIAgent",
    "STANDARD_GAME_PROMPT",
    "_detect_game_from_observation",
]
