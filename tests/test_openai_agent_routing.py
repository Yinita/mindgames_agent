from typing import Dict, Optional

import pytest

try:
    from external.mindgames_agent.src.agents.agent import OpenAIAgent
except ImportError:  # pragma: no cover - run within package without external prefix
    from agents.agent import OpenAIAgent  # type: ignore

try:
    from external.mindgames_agent.src.utils.action_normalizer import ActionNormalizer
except ImportError:  # pragma: no cover - run within package without external prefix
    from utils.action_normalizer import ActionNormalizer  # type: ignore


def test_inject_thinking_flag():
    base_kwargs: Dict[str, Dict[str, Dict[str, bool]]] = {}
    updated = OpenAIAgent._inject_thinking_flag(base_kwargs, True)
    assert updated["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    base_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    updated = OpenAIAgent._inject_thinking_flag(base_kwargs, True)
    assert updated["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    untouched = OpenAIAgent._inject_thinking_flag({}, False)
    assert "extra_body" not in untouched


def test_resolve_system_prompt_with_routing():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent.prompt_variant = None
    agent._prompt_loader = lambda game, variant: f"prompt:{game}"

    routed_prompt = agent._resolve_system_prompt("[GAME] Welcome to Secret Mafia! Night phase...")
    assert routed_prompt == "prompt:secret_mafia"

    agent._prompt_loader = None
    assert agent._resolve_system_prompt("anything") == "fallback"

    agent._static_system_prompt = "custom"
    assert agent._resolve_system_prompt("ignored observation") == "custom"

def test_call_returns_error_on_exception():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Completions:
        def create(self, *args, **kwargs):
            raise RuntimeError("boom")

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    with pytest.raises(RuntimeError):
        agent("observation")


def test_call_returns_error_on_empty_content():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Message:
        content = ""

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, *args, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    assert agent("observation") == "[ERROR]"


def test_call_strips_think_block():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Message:
        def __init__(self):
            self.content = "<think>reasoning</think>[A1 B1]"

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, *args, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    assert agent("observation") == "[A1 B1]"


def test_resolve_system_prompt_colonel_override():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent.model_name = "dummy"
    agent._completion_kwargs = {}
    agent.prompt_variant = None

    def _loader(game: str, variant: Optional[str]) -> Optional[str]:
        assert game == "colonel_blotto"
        return "Rules... Format: '[A4 B2 C2]' End"

    agent._prompt_loader = _loader

    class _Client:
        def __init__(self):
            self.chat = None

    agent._client = _Client()  # not used for prompt resolution

    prompt = agent._resolve_system_prompt("[GAME] Colonel Blotto ...")
    assert "Format: '[A4 B2 C2]'" in prompt


def test_normalize_observation_colonel_override():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    colonel_obs = (
        "[GAME] ColonelBlotto... Format: '[A4 B2 C2]' some text\n"
        "Format: '[A4 B2 C2]'"
    )
    normalized = agent._normalize_observation(colonel_obs)
    assert normalized == colonel_obs


def test_colonel_numeric_format_replacement():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    numeric_obs = "Format: '[4,2,2]' some other instructions"
    normalized = agent._normalize_observation(numeric_obs)
    assert normalized == numeric_obs

    def _loader(game: str, variant: Optional[str]) -> Optional[str]:
        return "Format: '[4, 2, 2]' rules"

    agent._prompt_loader = _loader
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    prompt = agent._resolve_system_prompt("[GAME] Colonel Blotto ...")
    assert "[4, 2, 2]" in prompt


def test_call_handles_structured_content_list():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Message:
        def __init__(self):
            self.content = [{"type": "text", "text": "<think>plan</think>[pass]"}]

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, *args, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    assert agent("observation") == "[pass]"


def test_call_handles_reasoning_output_text():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Message:
        def __init__(self):
            self.content = None
            self.reasoning = {"output_text": "[pass]"}

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, *args, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    assert agent("observation") == "[pass]"


def test_colonel_normalizer_filters_extra_fields():
    normalizer = ActionNormalizer.instance()
    observation = "[GAME] Colonel Blotto\nFormat: '[A4 B2 C2]'"
    result = normalizer.normalize_from_observation(
        "colonel_blotto", observation, "[D1 E0 T20 A8 B7 C5]"
    )
    assert result == "[A8 B7 C5]"


def test_call_handles_choice_reasoning_content():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model_name = "dummy"
    agent._static_system_prompt = None
    agent._fallback_system_prompt = "fallback"
    agent._prompt_loader = None
    agent._completion_kwargs = {}

    class _Message:
        def __init__(self):
            self.content = None

    class _Choice:
        def __init__(self):
            self.message = _Message()
            self.reasoning_content = "\n[pass]"

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        def create(self, *args, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    agent._client = _Client()

    assert agent("observation") == "[pass]"
