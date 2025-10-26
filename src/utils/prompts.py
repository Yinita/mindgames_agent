from __future__ import annotations
"""
Prompt utilities.

This module provides helpers to load per-game system prompts from either:
- Text files under `configs/prompts/<game>/default.txt`, or
- JSON mapping `configs/system_prompts.json`.

Note: Battle runs use a simplified selection (see ModelFactory in
`src/small_model/battle_system.py`): prefer `ModelConfig.system_prompt`,
then per-game `default.txt`, then JSON, and otherwise a built-in default.
"""

import json
from pathlib import Path
from typing import Dict, Optional


_CACHE: Optional[Dict[str, str]] = None


def _repo_root() -> Path:
    """Locate the nearest project root that provides configs/prompts."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        prompts_dir = parent / "configs" / "prompts"
        if prompts_dir.exists():
            return parent
    # Fallback to the original local repo structure (external/mindgames_agent)
    return current.parents[2]


def _default_prompts_path() -> Path:
    return _repo_root() / "configs" / "system_prompts.json"


def _prompt_dir() -> Path:
    return _repo_root() / "configs" / "prompts"


def _load_prompt_from_fs(game: str, variant: str = "default") -> Optional[str]:
    """Attempt to load a prompt snippet from configs/prompts/<game>/<variant>.txt."""
    prompt_path = _prompt_dir() / game / f"{variant}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return None


def load_system_prompts(path: Optional[str | Path] = None) -> Dict[str, str]:
    """Load per-game system prompts from JSON mapping.

    Expected format: {"colonel_blotto": "...", "three_player_ipd": "...", ...}
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    cfg_path = Path(path) if path else _default_prompts_path()
    if not cfg_path.exists():
        _CACHE = {}
        return _CACHE

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        data = {}
    # normalise keys to lower snake
    norm: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(v, str):
            continue
        kk = (k or "").strip().lower().replace("-v0", "").replace(" ", "_")
        norm[kk] = v
    _CACHE = norm
    return _CACHE


def get_game_prompt(game: str, variant: Optional[str] = None) -> Optional[str]:
    """Return the system prompt for a given game.

    The optional ``variant`` parameter is retained for backwards compatibility but
    is currently ignored; only the default prompt is returned when available.
    """
    key = (game or "").strip().lower().replace("-v0", "").replace(" ", "_")
    prompt_from_fs = _load_prompt_from_fs(key, "default")
    if prompt_from_fs:
        return prompt_from_fs

    data = load_system_prompts()
    return data.get(key)


__all__ = ["load_system_prompts", "get_game_prompt"]
