from __future__ import annotations

from typing import List

try:
    import external.mindgames_agent.src.main as entry
except ImportError:  # pragma: no cover - run within package without external prefix
    import main as entry  # type: ignore


class _DummyAgent:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, observation: str) -> str:
        self.calls.append(observation)
        return "ACTION"


def test_cli_main(monkeypatch, capsys):
    dummy = _DummyAgent()
    monkeypatch.setattr(entry, "agent_factory", lambda *args, **kwargs: dummy)

    entry.main(["--observation", "hello world"])

    captured = capsys.readouterr()
    assert captured.out.strip() == "ACTION"
    assert dummy.calls == ["hello world"]
