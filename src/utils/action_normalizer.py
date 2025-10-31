import re
from typing import Dict, List, Optional

# ---- Shared helpers for parsing [GAME] instructions ----
_LAST_GAME_TAG = re.compile(r"\[GAME\]", re.I)
_TAD_ON_YOUR_TURN = "On your turn, simply type your message."


def extract_truth_and_deception_instruction(observation: Optional[str]) -> Optional[str]:
    """TruthAndDeception rule:
    - If fewer than two [GAME]: take from the first [GAME] to the sentence
      "On your turn, simply type your message." inclusive.
    - If two or more [GAME]: take from the second [GAME] to the end.
    Mirrors src/log_analysis/parsers.py for consistency at inference time.
    """
    if not observation or not isinstance(observation, str):
        return None

    # Find all [GAME] occurrences
    matches = list(_LAST_GAME_TAG.finditer(observation))
    if not matches:
        return None

    if len(matches) < 2:
        start = matches[0].end()
        tail = observation[start:]
        # Up to the target sentence inclusive
        idx = tail.find(_TAD_ON_YOUR_TURN)
        if idx != -1:
            end = idx + len(_TAD_ON_YOUR_TURN)
            chunk = tail[:end]
        else:
            # Fallback: entire remainder
            chunk = tail
        return chunk.lstrip("\r\n ").rstrip()
    else:
        start = matches[1].end()
        chunk = observation[start:]
        return chunk.lstrip("\r\n ").rstrip()


def extract_last_game_line(observation: Optional[str]) -> Optional[str]:
    """Return the content of the last [GAME] line if present."""
    if not observation or not isinstance(observation, str):
        return None
    game_lines = re.findall(r"^\s*\[GAME\]\s*(.*)$", observation, flags=re.I | re.M)
    return game_lines[-1] if game_lines else None


class ActionNormalizer:
    _instance = None

    @staticmethod
    def instance() -> "ActionNormalizer":
        if ActionNormalizer._instance is None:
            ActionNormalizer._instance = ActionNormalizer()
        return ActionNormalizer._instance

    # ---- Public API ----
    def normalize(self, game: str, raw_action: str, context: Optional[Dict] = None) -> str:
        """
        Normalize raw action text to the strict format expected by each env.

        Args:
            game: One of ["Codenames", "ColonelBlotto", "SecretMafia", "ThreePlayerIPD"].
            raw_action: Free-form action text from agent.
            context: Optional hints to improve normalization.
                - Codenames:
                    { "role": "spymaster"|"operative", "board_words": List[str] }
                - ColonelBlotto:
                    { "fields": List[str], "num_total_units": int }
                - SecretMafia:
                    { "alive_players": List[int] }  # to clamp target
                - ThreePlayerIPD:
                    { "phase": "conversation"|"decision", "opponents": List[int] }
        Returns:
            The normalized action string.
        """
        context = context or {}
        game = self._canonical_game_name(game)
        if game == "codenames":
            return self._codenames(raw_action, context)
        if game == "colonel_blotto":
            return self._colonel_blotto(raw_action, context)
        if game == "secret_mafia":
            return self._secret_mafia(raw_action, context)
        if game == "three_player_ipd":
            return self._three_player_ipd(raw_action, context)
        if game == "truth_and_deception":
            return self._truth_and_deception(raw_action, context)
        # default: return as-is
        return raw_action

    def normalize_from_observation(self, game: str, observation: Optional[str], raw_action: str) -> str:
        """
        Convenience API: infer context from observation text so callers only need obs + action.

        Args:
            game: Game name (case-insensitive).
            observation: Full observation text that contains `[GAME]` lines.
            raw_action: Free-form action text from agent.
        Returns:
            Normalized action string suitable for the env.
        """
        canonical = self._canonical_game_name(game)
        ctx = self._infer_context_from_observation(canonical, observation or "")
        return self.normalize(game, raw_action, ctx)

    def _canonical_game_name(self, game: Optional[str]) -> str:
        """Normalize various game identifiers to canonical snake_case keys."""
        if not game:
            return ""
        key = (game or "").strip().lower()
        key = key.replace("-v0", "")
        key = key.replace("-v1", "")
        key = key.replace("-", "_").replace(" ", "_")
        if key == "threeplayeripd":
            key = "three_player_ipd"
        elif key == "secretmafia":
            key = "secret_mafia"
        elif key == "colonelblotto":
            key = "colonel_blotto"
        elif key == "truthanddeception":
            key = "truth_and_deception"
        return key

    # ---- Codenames ----
    def _codenames(self, text: str, ctx: Dict) -> str:
        s = (text or "").strip()
        if not s:
            return "[pass]"

        # Remove think tags and collapse whitespace for easier parsing.
        s = re.sub(r"<think>.*?</think>", " ", s, flags=re.I | re.S)
        s = re.sub(r"\s+", " ", s).strip()

        role = (ctx.get("role") or "").lower()

        # Spymaster format: [word number]
        # Accept inputs like: "wind 2", "[wind 2]", "word: wind, number: 2" etc.
        if role == "spymaster":
            # Prefer bracketed pairs, but avoid meta tokens like [Player 0]
            candidates = [(mm.group(1), mm.group(2)) for mm in re.finditer(r"\[([A-Za-z]+)\s+(\d+)\]", s)]
            if candidates:
                stop = {"player", "team", "red", "blue"}
                for word, num in candidates:
                    if word.lower() not in stop:
                        return f"[{word.lower()} {int(num)}]"
                # fallback: use the last pair if all were filtered
                word, num = candidates[-1]
                return f"[{word.lower()} {int(num)}]"
            # fallback extract any word + number
            m2 = re.search(r"(\w+)[^\d]*(\d+)", s)
            if m2:
                word, num = m2.group(1), m2.group(2)
                return f"[{word.lower()} {int(num)}]"
            # last resort: keep original wrapped to trigger env validation
            token = re.sub(r"\s+", " ", s)
            return f"[{token}]" if not s.startswith("[") else s

        # Operative format: [word] or [pass]
        # Allow plain word/pass, with or without brackets
        tokens = re.findall(r"\[\s*([A-Za-z]+)\s*\]", s)
        if tokens:
            return " ".join(f"[{tok.lower()}]" for tok in tokens)
        # extract first single token like 'pass' or word
        m3 = re.search(r"\b(pass|[a-zA-Z]+)\b", s, re.I)
        if m3:
            return f"[{m3.group(1).lower()}]"
        return "[pass]"  # safe default ends guessing

    # ---- Colonel Blotto ----
    def _colonel_blotto(self, text: str, ctx: Dict) -> str:
        s = (text or "").strip()
        compact_digits = re.sub(r"[^\d]", "", s)
        digit_groups = re.findall(r"\d+", s)
        if len(compact_digits) == 3 and digit_groups and len(digit_groups) == 1 and len(digit_groups[0]) == 3:
            fields_ctx: List[str] = [f.upper() for f in (ctx.get("fields") or [])]
            labels = fields_ctx[:3] if len(fields_ctx) >= 3 else ["A", "B", "C"]
            digits = [int(ch) for ch in compact_digits]
            paired = [f"{labels[i]}{digits[i]}" for i in range(min(3, len(labels), len(digits)))]
            if paired:
                return f"[{' '.join(paired)}]"
        token_re = re.compile(r"([A-Za-z])\s*[:=]?\s*(\d+)")
        parsed_pairs = [(m.group(1).upper(), int(m.group(2))) for m in token_re.finditer(s)]

        if not parsed_pairs:
            fields: List[str] = [f for f in (ctx.get("fields") or [])]
            nums = list(map(int, re.findall(r"\d+", s))) if s else []
            if fields and nums:
                parsed_pairs = [(f.upper(), n) for f, n in zip(fields, nums)]

        if not parsed_pairs:
            # As a last resort, return empty allocation brackets to trigger invalid handling upstream
            return text

        total_units = ctx.get("num_total_units")
        if isinstance(total_units, int):
            sum_units = sum(units for _, units in parsed_pairs)
            if sum_units > total_units:
                return text

        # If context provides fields, include zeros for omitted fields in stable order
        fields_ctx: List[str] = [f.upper() for f in (ctx.get("fields") or [])]
        if fields_ctx:
            matched_fields = {field for field, _ in parsed_pairs}
            if matched_fields == set(fields_ctx):
                if isinstance(total_units, int):
                    sum_units = sum(units for _, units in parsed_pairs)
                    if sum_units != total_units:
                        return text
                ordered = []
                for field in fields_ctx:
                    value = next((units for f, units in parsed_pairs if f == field), 0)
                    ordered.append(f"{field}{value}")
                return f"[{' '.join(ordered)}]"
        # Otherwise keep just the parsed pairs order
        parts = [f"{field}{units}" for field, units in parsed_pairs]
        return f"[{' '.join(parts)}]"

    # ---- Secret Mafia ----
    def _secret_mafia(self, text: str, ctx: Dict) -> str:
        s = (text or "").strip()
        phase = (ctx.get("phase") or "").lower()
        instruction = (ctx.get("instruction") or "").strip()

        # Pattern set mapping
        def requires_bracket_x(instr: str) -> bool:
            if not instr:
                return False
            if re.search(r"Night has fallen\. Mafia, agree on a victim\.", instr):
                return True
            if re.search(r"^Voting phase\s*-", instr, re.I):
                return True
            if re.search(r"^Night phase\s*-\s*choose one player", instr, re.I):
                return True
            if re.search(r"Player\s+\d+\s+attempted an invalid move", instr, re.I):
                return True
            if re.search(r"Player\s+\d+\s+has been eliminated", instr, re.I):
                return True
            return False

        def is_day_discuss(instr: str) -> bool:
            return bool(re.search(r"Day breaks\. Discuss for 3 rounds, then a vote will follow\.", instr))

        # Helper: choose a target X
        def choose_target() -> Optional[int]:
            # prefer explicit bracket or numeric in s
            m_player = re.search(r"\[(?:player\s*)?(\d+)\]", s, re.I)
            if m_player:
                return int(m_player.group(1))
            nums = list(map(int, re.findall(r"\b(\d+)\b", s)))
            alive: Optional[List[int]] = ctx.get("alive_players")
            self_id = ctx.get("self_id")
            negated: set[int] = set()
            neg_patterns = [
                r"(?:should\s+not|do\s+not|don't|not)\s+\w*\s*vote\s+for\s+player\s*(\d+)",
                r"(?:should\s+not|do\s+not|don't|not)\s+\w*\s*vote\s+for\s+(\d+)",
            ]
            for pat in neg_patterns:
                for match in re.findall(pat, s, re.I):
                    try:
                        negated.add(int(match))
                    except ValueError:
                        continue
            if alive:
                for x in nums:
                    if self_id is not None and x == self_id:
                        continue
                    if x in negated:
                        continue
                    if x in alive:
                        return x
                # fallback: first alive not self
                for x in alive:
                    if self_id is None or x != self_id:
                        return x
            # last resort: first numeric
            if nums:
                for x in nums:
                    if x not in negated:
                        return x
                return nums[0]
            return None

        # Day discussion: keep original text (minimal cleaning) per pattern
        if is_day_discuss(instruction) or phase == "day_discuss":
            return s if s else "ok"

        # For phases/instructions requiring an [X] token
        if requires_bracket_x(instruction) or phase in ("night_investigate", "vote"):
            target = choose_target()
            return f"[{target}]" if target is not None else s or text

        # Otherwise, reuse existing tolerant parsing for ad-hoc texts
        # Accept role-tagged
        m_tag = re.search(r"\[(?:Mafia|Investigate|Protect)\s*:\s*Player\s*(\d+)\]", s, re.I)
        if m_tag:
            return f"[{int(m_tag.group(1))}]"
        # Accept vote phrasing
        m_vote = re.search(r"\bVote\s+for\s+Player\s+(\d+)\b", s, re.I)
        if m_vote:
            return f"[{int(m_vote.group(1))}]"
        m_vote_br = re.search(r"\[\s*Vote\s+for\s+Player\s+(\d+)\s*\]", s, re.I)
        if m_vote_br:
            return f"[{int(m_vote_br.group(1))}]"
        # Accept simplified [Player X]
        m_player2 = re.search(r"\[(?:player\s*)?(\d+)\]", s, re.I)
        if m_player2:
            return f"[{int(m_player2.group(1))}]"
        return s or text

    # ---- Three Player IPD ----
    def _three_player_ipd(self, text: str, ctx: Dict) -> str:
        s = (text or "").strip()
        phase = (ctx.get("phase") or "").lower()
        instruction = (ctx.get("instruction") or "").strip()
        # Pattern mapping overrides phase when present
        is_chat_prompt = bool(re.search(r"^\s*───\s*Starting\s+Round", instruction)) or bool(re.search(r"You can converse freely for the next", instruction, re.I))
        is_decision_prompt = bool(re.search(r"Chat finished for round", instruction, re.I)) or bool(re.search(r"Submit your decisions", instruction, re.I))
        if is_chat_prompt:
            phase = "chat"
        elif is_decision_prompt:
            phase = "decision"
        if phase == "conversation" or phase == "chat":
            # In chat, strip any accidental decision tokens and return remaining chat
            s_clean = re.sub(r"\[\s*\d+\s+(?:cooperate|defect)\s*\]", "", s, flags=re.I)
            # Remove any stray decision labels like "Decision:" if appear
            s_clean = re.sub(r"\bDecision\s*:\s*", "", s_clean, flags=re.I)
            s_clean = re.sub(r"\s+", " ", s_clean).strip()
            return s_clean if s_clean else "ok"

        # Decision phase: tokens like [1 defect] [2 cooperate]
        # Parse any valid tokens first
        token_pat = re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.I)
        tokens = [(int(pid), choice.lower()) for pid, choice in token_pat.findall(s)]

        # If none found, try to infer from simple patterns like: "defect 1,2" or "cooperate 2"
        if not tokens:
            opps: List[int] = ctx.get("opponents") or []
            # Phrases like "cooperate with Player 1" or "defect Player 2"
            phrase_pat = re.compile(r"\b(cooperate|defect)\b[^\d]{0,20}?(?:with\s+)?player\s*(\d+)", re.I)
            for choice, pid in phrase_pat.findall(s):
                tokens.append((int(pid), choice.lower()))
            if not tokens:
                reverse_pat = re.compile(r"player\s*(\d+)[^\d]{0,20}?\b(cooperate|defect)\b", re.I)
                for pid, choice in reverse_pat.findall(s):
                    tokens.append((int(pid), choice.lower()))

        if not tokens:
            opps = ctx.get("opponents") or []
            # collect explicit mentions
            mentioned = list(map(int, re.findall(r"\b(\d+)\b", s)))
            # choice heuristic
            choice = "defect" if re.search(r"\bdefect\b", s, re.I) else ("cooperate" if re.search(r"\bcooperate\b", s, re.I) else None)
            if choice and (opps or mentioned):
                targets = mentioned or opps
                tokens = [(pid, choice) for pid in targets]

        # If still empty, default to cooperate with all known opponents if provided
        if not tokens:
            opps: List[int] = ctx.get("opponents") or []
            if opps:
                tokens = [(pid, "cooperate") for pid in opps]

        if not tokens:
            return text  # will be treated as no valid tokens; env defaults unspecified to cooperate after acting

        # Deduplicate keeping last decision per opponent
        final: Dict[int, str] = {}
        for pid, choice in tokens:
            final[pid] = "defect" if choice.startswith("d") else "cooperate"
        parts = [f"[{pid} {final[pid]}]" for pid in sorted(final)]
        return " ".join(parts)

    # ---- Truth and Deception ----
    def _truth_and_deception(self, text: str, ctx: Dict) -> str:
        instruction = ctx.get("instruction") or ""
        instruction = re.sub(r"\s+", " ", instruction).strip()
        s = re.sub(r"\s+", " ", (text or "").strip())
        if not instruction:
            return s
        if "truth" in instruction.lower() and "deception" in instruction.lower():
            # Let original text pass through to preserve nuance when unsure
            return s
        return s

    # ---- Observation parsers (context inference) ----
    def _infer_context_from_observation(self, game: str, obs: str) -> Dict:
        obs = obs or ""
        if game == "codenames":
            return self._infer_ctx_codenames(obs)
        if game == "colonelblotto" or game == "colonel_blotto":
            return self._infer_ctx_colonel_blotto(obs)
        if game == "secretmafia" or game == "secret_mafia":
            return self._infer_ctx_secret_mafia(obs)
        if game == "threeplayeripd" or game == "three_player_ipd":
            return self._infer_ctx_three_player_ipd(obs)
        if game == "truthanddeception" or game == "truth_and_deception":
            return self._infer_ctx_truth_and_deception(obs)
        return {}

    def _infer_ctx_codenames(self, obs: str) -> Dict:
        ctx: Dict = {}
        # Detect role from [GAME] role line
        if re.search(r"\[GAME\].*You are .*Spymaster", obs, re.I):
            ctx["role"] = "spymaster"
        elif re.search(r"\[GAME\].*You are .*Operative", obs, re.I):
            ctx["role"] = "operative"
        if "role" not in ctx:
            if re.search(r"You\s+are\s+.*Spymaster", obs, re.I):
                ctx["role"] = "spymaster"
            elif re.search(r"You\s+are\s+.*Operative", obs, re.I):
                ctx["role"] = "operative"
        # If spymaster submitted clue, it implies operative turn
        if not ctx.get("role") and re.search(r"\[GAME\].*Spymaster submitted clue", obs, re.I):
            ctx["role"] = "operative"
        # Attempt to extract board words if present (optional improvement)
        words = re.findall(r"\b([A-Za-z]{2,})\b", obs)
        blacklist = {
            "you",
            "are",
            "the",
            "your",
            "clue",
            "guess",
            "words",
            "board",
            "team",
            "spymaster",
            "operative",
            "round",
            "turn",
        }
        board_words = [w.lower() for w in words if w.lower() not in blacklist]
        if board_words:
            ctx["board_words"] = board_words
        return ctx

    def _infer_ctx_colonel_blotto(self, obs: str) -> Dict:
        ctx: Dict = {}
        match = re.search(r"Units to allocate:\s*(\d+)", obs, re.I)
        if match:
            ctx["num_total_units"] = int(match.group(1))
        fields = re.findall(r"fields?\s*:\s*([A-Za-z,\s]+)", obs, re.I)
        if fields:
            field_list = re.split(r"[\s,]+", fields[-1].strip())
            field_list = [f for f in field_list if f]
            ctx["fields"] = field_list
        else:
            match_fields = re.findall(r"Available fields:\s*([A-Za-z,\s]+)", obs, re.I)
            if match_fields:
                field_list = re.split(r"[\s,]+", match_fields[-1].strip())
                field_list = [f for f in field_list if f]
                ctx["fields"] = field_list
        return ctx

    def _infer_ctx_secret_mafia(self, obs: str) -> Dict:
        ctx: Dict = {}
        alive = list(map(int, re.findall(r"Alive players:\s*([0-9,\s]+)", obs, re.I)))
        if alive:
            ctx["alive_players"] = alive
        instruction = extract_last_game_line(obs)
        if instruction:
            ctx["instruction"] = instruction
        m_self = re.search(r"You are Player\s*(\d+)", obs, re.I)
        if m_self:
            ctx["self_id"] = int(m_self.group(1))
        return ctx

    def _infer_ctx_three_player_ipd(self, obs: str) -> Dict:
        ctx: Dict = {}
        m_phase = re.search(r"Phase:\s*(\w+)", obs, re.I)
        if m_phase:
            ctx["phase"] = m_phase.group(1).lower()
        instruction = extract_last_game_line(obs)
        if instruction:
            ctx["instruction"] = instruction
        opponents = [int(x) for x in re.findall(r"Opponents?:\s*([0-9,\s]+)", obs, re.I)]
        if opponents:
            ctx["opponents"] = opponents
        return ctx

    def _infer_ctx_truth_and_deception(self, obs: str) -> Dict:
        ctx: Dict = {}
        instruction = extract_truth_and_deception_instruction(obs)
        if instruction:
            ctx["instruction"] = instruction
        return ctx


__all__ = [
    "ActionNormalizer",
    "extract_last_game_line",
    "extract_truth_and_deception_instruction",
]
