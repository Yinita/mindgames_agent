from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
import re
from typing import Any, Callable, Dict, Optional

try:
    from ..utils.prompts import get_game_prompt  # type: ignore
except ImportError:  # pragma: no cover - package fallback
    try:
        from utils.prompts import get_game_prompt  # type: ignore
    except ImportError as fallback_import_error:
        raise ImportError("Failed to import get_game_prompt") from fallback_import_error

try:
    from ..utils.action_normalizer import ActionNormalizer  # type: ignore
except ImportError:  # pragma: no cover - package fallback
    try:
        from utils.action_normalizer import ActionNormalizer  # type: ignore
    except ImportError as fallback_import_error:
        raise ImportError("Failed to import ActionNormalizer") from fallback_import_error


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
    """Thin wrapper around the OpenAI-compatible vLLM deployment with automatic prompt routing."""

    DEFAULT_TIMEOUT = 300.0
    DEFAULT_MAX_TOKENS = 10000
    DEFAULT_MODEL_NAME = "yinita/mg-8b-cot-sft-general-1024"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
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
            prompt_variant: Optional hint passed to the prompt loader (reserved for future use).
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

        self.model_name = model_name or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL_NAME)
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

        resolved_base = base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
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
                prompt_variant = getattr(self, "prompt_variant", None)
                prompt = self._prompt_loader(game, prompt_variant)
                if prompt:
                    return prompt

        return self._fallback_system_prompt

    def _normalize_observation(self, observation: str) -> str:
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
        except Exception as exc:
            logging.getLogger(__name__).exception("Chat completion request failed")
            raise

        choice = response.choices[0]
        content = self._extract_choice_content(choice)
        if isinstance(content, str):
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
            if cleaned:
                normalized = self._normalize_action(normalized_observation, cleaned)
                if normalized:
                    return normalized
                action = self._extract_bracket_action(cleaned)
                return action or cleaned
            stripped = content.strip()
            if stripped:
                return stripped
        return ""

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
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer

    def _extract_choice_content(self, choice: Any) -> Optional[str]:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        message_content = None
        if message is not None:
            message_content = self._extract_message_content(message)
            if message_content:
                return message_content
        reasoning_content = self._coerce_reasoning_to_text(
            self._maybe_get(choice, "reasoning_content")
        )
        if reasoning_content:
            return reasoning_content
        return None

    @staticmethod
    def _extract_message_content(message: Any) -> Optional[str]:
        content = OpenAIAgent._coerce_content_to_text(
            OpenAIAgent._maybe_get(message, "content")
        )
        if content:
            return content

        for key in ("output_text", "final_answer", "final", "answer"):
            value = OpenAIAgent._maybe_get(message, key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        reasoning = OpenAIAgent._maybe_get(message, "reasoning")
        reasoning_text = OpenAIAgent._coerce_reasoning_to_text(reasoning)
        if reasoning_text:
            return reasoning_text

        return None

    @staticmethod
    def _maybe_get(container: Any, key: str) -> Any:
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(key)
        if hasattr(container, key):
            return getattr(container, key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_content_to_text(content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            stripped = content.strip()
            return stripped if stripped else None
        if isinstance(content, (list, tuple)):
            parts: list[str] = []
            for item in content:
                text = None
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                    if text is None and "children" in item:
                        text = OpenAIAgent._coerce_content_to_text(item.get("children"))
                    if text is None and "output_text" in item:
                        text = item.get("output_text")
                elif hasattr(item, "text"):
                    text = getattr(item, "text")
                if text is None:
                    # attempt nested extraction for complex objects
                    text = OpenAIAgent._coerce_content_to_text(OpenAIAgent._maybe_get(item, "text"))
                if isinstance(text, (list, tuple)):
                    text = OpenAIAgent._coerce_content_to_text(text)
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped:
                        parts.append(stripped)
            if parts:
                return "\n".join(parts)
            return None
        if isinstance(content, dict):
            return OpenAIAgent._coerce_content_to_text(list(content.values()))
        return None

    @staticmethod
    def _coerce_reasoning_to_text(reasoning: Any) -> Optional[str]:
        if reasoning is None:
            return None
        if isinstance(reasoning, str):
            stripped = reasoning.strip()
            return stripped if stripped else None
        if isinstance(reasoning, dict):
            for key in ("output_text", "answer", "final_answer", "final", "text", "content"):
                value = reasoning.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            combined: list[str] = []
            for key in ("reasoning_content", "steps", "thoughts", "messages"):
                value = reasoning.get(key)
                text = OpenAIAgent._coerce_content_to_text(value)
                if text:
                    combined.append(text)
            if combined:
                return "\n".join(part for part in combined if part)
            return None
        if isinstance(reasoning, (list, tuple)):
            collected: list[str] = []
            for item in reasoning:
                text = OpenAIAgent._coerce_reasoning_to_text(item)
                if text:
                    collected.append(text)
            if collected:
                return "\n".join(collected)
        return None

    def _normalize_action(self, observation: str, raw_action: str) -> Optional[str]:
        if not raw_action:
            return None
        if ActionNormalizer is None:
            return None
        game = _detect_game_from_observation(observation)
        if not game:
            return None
        try:
            normalized = ActionNormalizer.instance().normalize_from_observation(
                game, observation, raw_action
            )
        except Exception:
            return None
        if isinstance(normalized, str) and normalized.strip():
            return normalized.strip()
        return None

    @staticmethod
    def _extract_bracket_action(text: str) -> Optional[str]:
        """Best-effort extraction of a bracketed Colon Blotto action."""
        matches = re.findall(r"\[([^\]]+)\]", text)
        if not matches:
            return None

        for payload in reversed(matches):
            payload = payload.strip()
            if not any(ch.isdigit() for ch in payload):
                continue

            if re.fullmatch(r"\d{3}", payload):
                a, b, c = payload
                return f"[A{a} B{b} C{c}]"

            cleaned = payload.replace(",", " ")
            cleaned = re.sub(r"([A-Za-z])[ ]*:", r"\1", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
            tokens = cleaned.split()

            mapped: list[str] = []
            simple_digits: list[str] = []
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                if tok[0].isalpha():
                    label = tok[0]
                    value = re.sub(r"\D", "", tok[1:])
                    if value:
                        mapped.append(f"{label}{value}")
                elif tok.isdigit():
                    simple_digits.append(tok)

            if mapped and mapped == tokens[:len(mapped)]:
                return "[" + " ".join(mapped) + "]"

            if len(simple_digits) == 3:
                labels = ["A", "B", "C"]
                combined = [f"{labels[i]}{val}" for i, val in enumerate(simple_digits)]
                return "[" + " ".join(combined) + "]"

            if mapped:
                return "[" + " ".join(mapped) + "]"

        return None

__all__ = [
    "Agent",
    "OpenAIAgent",
    "STANDARD_GAME_PROMPT",
    "_detect_game_from_observation",
]
