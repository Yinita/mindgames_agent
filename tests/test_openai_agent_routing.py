from typing import Dict, Optional

from src.agents.agent import OpenAIAgent


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


def test_inject_thinking_flag():
    base_kwargs: Dict[str, Dict[str, Dict[str, bool]]] = {}
    updated = OpenAIAgent._inject_thinking_flag(base_kwargs, True)
    assert updated["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    # Ensure existing flags are preserved
    base_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    updated = OpenAIAgent._inject_thinking_flag(base_kwargs, True)
    assert updated["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    # Disabling thinking leaves kwargs untouched
    untouched = OpenAIAgent._inject_thinking_flag({}, False)
    assert "extra_body" not in untouched


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

    assert agent("observation") == "[ERROR]"


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
    assert "Format: '[A7 B7 C6]'" in prompt
    assert "A4 B2 C2" not in prompt


def test_normalize_observation_colonel_override():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    colonel_obs = (
        "[GAME] ColonelBlotto... Format: '[A4 B2 C2]' some text\n"
        "Format: '[A4 B2 C2]'"
    )
    normalized = agent._normalize_observation(colonel_obs)
    assert "Format: '[A7 B7 C6]'" in normalized
    assert "A4 B2 C2" not in normalized
