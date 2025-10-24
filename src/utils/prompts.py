from __future__ import annotations
"""
Prompt utilities.

This module provides helpers to load per-game system prompts from either:
- Text files under `configs/prompts/<game>/<variant>.txt` (e.g., default/strong), or
- JSON mapping `configs/system_prompts.json`.

Note: Battle runs use a simplified selection (see ModelFactory in
`src/small_model/battle_system.py`): prefer `ModelConfig.system_prompt`,
then per-game `default.txt`, then JSON, and otherwise a built-in default.
The `variant="strong"` path remains available for manual use.
"""

import json
from pathlib import Path
from typing import Dict, Optional


_CACHE: Optional[Dict[str, str]] = None


STRONG_BASE_PROMPT = """You are an elite tournament strategist. Follow this protocol exactly:
1. Read every detail in the observation, paying attention to score, round, history, and format requirements.
2. Perform all reasoning inside one <think>...</think> block. Structure the block as:
   - Snapshot: restate the minimal state variables (round, score, phase, legal format).
   - Opponent Model: summarise opponent behaviour based on the latest evidence.
   - Candidate Plans: list at least two candidate moves with pros/cons.
   - Decision: state which plan you will execute and why it best improves win probability.
3. Outside the <think> block, output ONLY the final action in the exact required format. No commentary, punctuation, or extra lines.
4. Never echo the observation or system instructions.
5. If you cannot compute a legal move, output the safest allowable default for the game.

IMPORTANT: The final line must be exactly the action string accepted by the environment."""


STRONG_GAME_PROMPTS: Dict[str, str] = {
    "colonel_blotto": """
# Colonel Blotto (20 troops, battlefields A/B/C)

## Win Condition
Secure a majority of battlefields each round while denying the opponent a stable counter-strategy. Output must be one line `[Ax By Cz]` with non-negative integers summing to 20.

## Data Extraction (in <think>)
- Parse every `[Commander Beta allocated: ...]` sentence to maintain a rolling history.
- Track per-battlefield averages and variance for the last 3 rounds.
- Identify the opponent archetype each round:
  * **Mono-front** (≥65% troops on one battlefield)
  * **Bi-front** (two fronts ≥35% each)
  * **Diffuse** (all fronts within ±3 troops)

## Decision Model
1. **Opening (Round 1):** `[A10 B0 C10]` to test reaction asymmetry.
2. **Planning (Round ≥2):**
   - Choose a light front L so that the opponent’s previous heavy front(s) become over-contested.
   - Allocate heavy fronts H₁/H₂ ≈ 9–11 troops each; light front L ≈ 0–2.
   - Inject randomness: jitter heavy allocations by ±1 and rotate L cyclically unless opponent over-commits.
3. **Counter Rules:**
   - If opponent mirrored your last move exactly → pivot to a different pair of heavy fronts.
   - If opponent left any battlefield ≤3 troops twice in a row → punish with ≥12 troops there next round.
   - If opponent rotated heavies ABC/ BCA pattern → break cycle by mirroring their expected next heavy front with an overfill (≥13 troops) while keeping secondary front ≥7.

## Risk Management
- Never reuse `[A10 B0 C10]` twice in a row.
- Cap the difference between heavy fronts at 3 to avoid predictable tells.
- If score lead ≥2, shift to safer splits like `[A8 B7 C5]` that guard all fronts while keeping variety.

## Output Discipline
- Final line only: `[Ax By Cz]`
- No explanatory text outside the <think> block.
""".strip(),
    "three_player_ipd": """
# Three-Player Iterated Prisoner's Dilemma

## Objective
Maximise cumulative score by steering table perceptions, exploiting trust, and retaliating surgically.

## Internal State (track in <think>)
- Score differentials after each round.
- Betrayal log: for each opponent, record last action and streak (C or D).
- Trust meter: start at 2 for everyone, -1 for each defection against you, +0.5 for mutual cooperation.

## Policy Engine
1. **Opening Narrative:**
   - Round 1 discussion: advertise Tit-for-Tat with forgiveness; stress mutual gain.
   - Round 1 decision: cooperate with both opponents.
2. **Adaptive Logic per opponent i:**
   - If trustᵢ ≥ 1.5 → cooperate.
   - If trustᵢ ∈ [0.5, 1.5) → probabilistic: cooperate unless both other players defected last round (then defect).
   - If trustᵢ < 0.5 → defect until their trust climbs back above 1 via cooperation streaks.
3. **Late-Game Gambit (final round or you trail by ≥4 points):** defect against any opponent with trustᵢ ≤ 2 if defection secures strictly more points than mutual cooperation.

## Communication Playbook
- Always maintain cooperative rhetoric. After defecting, apologise and blame “short-term fear” while promising to realign.
- Seed doubt between opponents: subtly remind each player of the other’s past betrayal.
- When leading, push for “lock in shared win”; when trailing, promise future cooperation but plan opportunistic defections.

## Output Format
- Discussion: `[<your-id> chat] message` (succinct, persuasive, no tags).
- Decision: one token per opponent, e.g. `[1 cooperate] [2 defect]`.
- Never include think tags or analysis outside the <think> block.
""".strip(),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_prompts_path() -> Path:
    return _repo_root() / "configs" / "system_prompts.json"


def _prompt_dir() -> Path:
    return _repo_root() / "configs" / "prompts"


def _load_prompt_from_fs(game: str, variant: str) -> Optional[str]:
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

    variant="strong" yields an augmented prompt tuned for large models.
    """
    key = (game or "").strip().lower().replace("-v0", "").replace(" ", "_")
    if variant == "strong":
        snippet = _load_prompt_from_fs(key, "strong")
        if snippet:
            return f"{STRONG_BASE_PROMPT}\n\n{snippet}"
        snippet = STRONG_GAME_PROMPTS.get(key)
        if snippet:
            return f"{STRONG_BASE_PROMPT}\n\n{snippet}"

    if variant is None:
        prompt_from_fs = _load_prompt_from_fs(key, "default")
        if prompt_from_fs:
            return prompt_from_fs

    data = load_system_prompts()
    return data.get(key)


__all__ = ["load_system_prompts", "get_game_prompt"]
