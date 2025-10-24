# Agent Guidelines

This submission branch keeps only the components required to run language-model
agents during evaluation. A quick refresher on what remains and how to work with it:

- `src/agent.py` exports the public API (`agent()` factory, `OpenAIAgent`, and `STANDARD_GAME_PROMPT`).
- `src/agents/agent.py` defines the lightweight agent base class and the OpenAI-backed
  implementation with automatic prompt routing.
- Prompt assets live in `configs/prompts/` (one subdirectory per game).
- Use `src/utils/prompts.py` to resolve prompts when you truly need a manual override.

## Build & Test

```bash
pip install -e .
pip install -e .[dev]  # optional: pytest
pytest
```

## Prompt Usage

```python
from src.utils.prompts import get_game_prompt

prompt = get_game_prompt("secret_mafia")
```

> Recommended: lean on the built-in router (`agent("openai")`) so matches always use our curated defaults. Only supply a custom prompt when you have a strong reason, then document it.

Keep new strategies self-contained by adding prompt files under `configs/prompts/<game>/`.

### CLI Entry

For automated evaluation, call the bundled CLI:

```bash
python -m src.main --model-name gpt-4o --observation "[GAME] ..."
```

Options mirror the Python API (`--backend`, `--system-prompt`, etc.).

If you skip `system_prompt` when constructing `OpenAIAgent`, it will attempt to detect
the game from each observation and pull the matching default prompt automatically,
falling back to the built-in standard prompt when no match is found.
