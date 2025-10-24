from typing import Any, Optional

from src.agents.agent import Agent, OpenAIAgent, STANDARD_GAME_PROMPT


def agent(
    name: str = "openai",
    *,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Agent:
    """Factory helper returning a callable agent instance.

    Usage:
        from src.agent import agent
        bot = agent("openai", model_name="gpt-4o")
        action = bot(observation)

    Args:
        name: Identifier for the backend. Currently only ``"openai"`` is supported.
        model_name: Optional override for the OpenAI model identifier.
        **kwargs: Forwarded to ``OpenAIAgent`` (e.g., system_prompt, api_key).
    """
    key = (name or "openai").strip().lower()
    if key in {"openai", "openai_agent", "default"}:
        return OpenAIAgent(model_name=model_name, **kwargs)

    raise ValueError(
        f"Unknown agent backend '{name}'. Supported options: 'openai'."
    )

__all__ = [
    "Agent",
    "OpenAIAgent",
    "STANDARD_GAME_PROMPT",
    "agent",
]
