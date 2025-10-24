import types

import pytest

import src.agent as agent_module


def test_agent_factory_returns_openai(monkeypatch):
    created = {}

    class DummyAgent:
        def __init__(self, *, model_name=None, **kwargs):
            created["model_name"] = model_name
            created["kwargs"] = kwargs

        def __call__(self, observation):
            return observation

    monkeypatch.setattr(agent_module, "OpenAIAgent", DummyAgent)

    bot = agent_module.agent("openai", model_name="gpt-test", foo="bar")
    assert isinstance(bot, DummyAgent)
    assert created["model_name"] == "gpt-test"
    assert created["kwargs"] == {"foo": "bar"}
    assert bot("OBS") == "OBS"


def test_agent_factory_unknown_backend():
    with pytest.raises(ValueError):
        agent_module.agent("unknown")
