from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.agent import agent as agent_factory


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise SystemExit(f"Observation file not found: {path}") from exc


def main(argv: Optional[list[str]] = None) -> str:
    parser = argparse.ArgumentParser(
        description="MindGames agent entry point. Returns a single action."
    )
    parser.add_argument(
        "--backend",
        default="openai",
        help="Agent backend identifier (default: openai).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier for the backend (e.g., gpt-4o).",
    )
    parser.add_argument(
        "--observation",
        default=None,
        help="Raw observation string. Mutually exclusive with --observation-file.",
    )
    parser.add_argument(
        "--observation-file",
        default=None,
        help="Path to a file containing the observation text.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional override for the system prompt passed to the agent.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a file whose contents will be used as the system prompt.",
    )

    args = parser.parse_args(argv)

    if args.observation and args.observation_file:
        parser.error("Specify either --observation or --observation-file, not both.")
    if not args.observation and not args.observation_file:
        parser.error("Provide --observation or --observation-file.")

    if args.observation:
        observation = args.observation
    else:
        observation = _read_text(Path(args.observation_file))

    system_prompt: Optional[str] = args.system_prompt
    if args.system_prompt_file:
        system_prompt = _read_text(Path(args.system_prompt_file))

    agent_kwargs = {}
    if system_prompt is not None:
        agent_kwargs["system_prompt"] = system_prompt

    bot = agent_factory(args.backend, model_name=args.model_name, **agent_kwargs)
    action = bot(observation)
    print(action)
    return action


if __name__ == "__main__":
    main()
