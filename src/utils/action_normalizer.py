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
            # last resort: keep original wrapped to trigger invalid handling
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
        fields_ctx: List[str] = [f.upper() for f in (ctx.get("fields") or [])]
        if not s:
            return text

        def format_pairs(pairs: List[tuple[str, int]]) -> Optional[str]:
            if not pairs:
                return None
            ordered: List[str] = []
            if fields_ctx:
                seen: Dict[str, int] = {}
                for field, units in pairs:
                    if field in fields_ctx and field not in seen:
                        seen[field] = units
                ordered = [f"{field}{seen[field]}" for field in fields_ctx if field in seen]
                if ordered:
                    return "[" + " ".join(ordered) + "]"
            ordered = [f"{field}{units}" for field, units in pairs]
            if ordered:
                return "[" + " ".join(ordered) + "]"
            return None

        # Prioritize the last bracketed allocation containing numeric tokens.
        bracket_segments = re.findall(r"\[([^\]]+)\]", s)
        for payload in reversed(bracket_segments):
            pairs = [
                (m.group(1).upper(), int(m.group(2)))
                for m in re.finditer(r"([A-Za-z])\s*[:=]?\s*(\d+)", payload)
            ]
            formatted = format_pairs(pairs)
            if formatted:
                return formatted

        # Fallback to scanning the entire text.
        pairs = [
            (m.group(1).upper(), int(m.group(2)))
            for m in re.finditer(r"([A-Za-z])\s*[:=]?\s*(\d+)", s)
        ]
        formatted = format_pairs(pairs)
        if formatted:
            return formatted

        return text

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
        """
        Chat turns: return lightweight cleaned chat (no [GAME] lines).
        Guess turn: normalize to "[Fact 1]" or "[Fact 2]" from flexible inputs.
        """
        s = (text or "").strip()
        # strip wrapping quotes frequently produced by models
        s = s.strip("\"'")
        phase = (ctx.get("phase") or "").lower()

        # During chat: ignore instructions and meta, keep concise text
        if phase in ("chat", "conversation", ""):
            s_clean = re.sub(r"\[GAME\][^\n]*", "", s, flags=re.I)
            s_clean = re.sub(r"\s+", " ", s_clean).strip()
            return s_clean if s_clean else "ok"

        # Guess phase: robustly map variants to [Fact 1] / [Fact 2]
        # 1) Exact bracketed token
        m = re.search(r"\[\s*Fact\s*(1|2)\s*\]", s, re.I)
        if m:
            return f"[Fact {int(m.group(1))}]"
        # 2) Simple numeric bracket like [1] / [2]
        m = re.search(r"\[\s*(1|2)\s*\]", s)
        if m:
            return f"[Fact {int(m.group(1))}]"
        # 3) Inline 'Fact 1' / 'Fact2'
        m = re.search(r"\bfact\s*(1|2)\b", s, re.I)
        if m:
            return f"[Fact {int(m.group(1))}]"
        # 4) Words like 'first'/'second', 'one'/'two'
        m = re.search(r"\b(first|second|one|two|1|2)\b", s, re.I)
        if m:
            tok = m.group(1).lower()
            if tok in ("2", "second", "two"):
                return "[Fact 2]"
            return "[Fact 1]"
        # 5) Fallback: any standalone 1 or 2 elsewhere
        nums = re.findall(r"\b(1|2)\b", s)
        if nums:
            return f"[Fact {int(nums[-1])}]"  # take last stated
        # If nothing matches, return empty to let upstream handle invalid
        return text

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
        # e.g., lines like: "Words: word1, word2, ..." or bracketed on board
        words = re.findall(r"\b([A-Za-z]{2,})\b", obs)
        # filter obvious non-words:
        common_stop = {"the", "and", "you", "are", "your", "team", "blue", "red", "neutral", "assassin", "round", "game", "chat", "guess", "clue", "submit", "spymaster", "operative"}
        board_words: List[str] = [w.lower() for w in words if w.isalpha() and w.lower() not in common_stop]
        if board_words:
            ctx["board_words"] = board_words[:50]  # cap to avoid bloat
        return ctx

    def _infer_ctx_colonel_blotto(self, obs: str) -> Dict:
        ctx: Dict = {}
        # Extract available fields from a line like: "Available fields: A, B, C"
        m = re.search(r"Available\s+fields:\s*([A-Za-z](?:\s*,\s*[A-Za-z])*)", obs)
        if m:
            fields = [t.strip().upper() for t in m.group(1).split(",") if t.strip()]
            if fields:
                ctx["fields"] = fields
        # Units to allocate: 20
        m2 = re.search(r"Units\s+to\s+allocate:\s*(\d+)", obs, re.I)
        if m2:
            ctx["num_total_units"] = int(m2.group(1))
        # Also infer fields from format hints like [A4 B2 C2]
        if "fields" not in ctx:
            fmt_fields = re.findall(r"\b([A-Za-z])\s*\d+\b", obs)
            if fmt_fields:
                ctx["fields"] = sorted(list({f.upper() for f in fmt_fields}))
        return ctx

    def _infer_ctx_secret_mafia(self, obs: str) -> Dict:
        ctx: Dict = {}
        # Players line: "Players: Player 0, Player 1, ..."
        players_line = re.search(r"Players:\s*([^\n]+)", obs)
        if players_line:
            ids = list(map(int, re.findall(r"\b(\d+)\b", players_line.group(1))))
            if ids:
                ctx["alive_players"] = ids
        # Self ID if present
        m_self = re.search(r"You\s+are\s+Player\s+(\d+)", obs, re.I)
        if m_self:
            ctx["self_id"] = int(m_self.group(1))
        # Phase inference based on the LAST [GAME] line
        game_lines = re.findall(r"^\s*\[GAME\]\s*(.*)$", obs, flags=re.I | re.M)
        last_instr = game_lines[-1] if game_lines else ""
        phase: Optional[str] = None
        for gl in reversed(game_lines):
            if re.search(r"Voting begins|Voting phase|cast your vote", gl, re.I):
                phase = "vote"
                break
            if re.search(r"Night phase\s*-\s*choose one player to investigate|Night phase.*investigate", gl, re.I):
                phase = "night_investigate"
                break
            if re.search(r"Day .*Discussion|Day breaks", gl, re.I):
                phase = "day_discuss"
                break
        if phase:
            ctx["phase"] = phase
        if last_instr:
            ctx["instruction"] = last_instr
        # If elimination updates are present, try to capture remaining numbers in relevant [GAME] lines
        if "alive_players" not in ctx:
            ids = list(map(int, re.findall(r"\[(\d+)\]", obs)))
            if ids:
                # heuristic: targets listed are usually alive candidates
                ctx["alive_players"] = sorted(list({i for i in ids}))
        return ctx

    def _infer_ctx_three_player_ipd(self, obs: str) -> Dict:
        ctx: Dict = {}
        # Phase: if chat finished prompt exists => decision; else chat
        if re.search(r"\[GAME\].*Chat finished.*Submit your decisions", obs, re.I):
            ctx["phase"] = "decision"
        else:
            # If explicit submit decisions exists, also decision
            if re.search(r"\[GAME\].*Submit your decisions", obs, re.I):
                ctx["phase"] = "decision"
            else:
                ctx["phase"] = "chat"
        # include last [GAME] line for pattern mapping
        last_instr = extract_last_game_line(obs)
        if last_instr:
            ctx["instruction"] = last_instr
        # Opponents: extract self id and player ids from text
        self_id = None
        m_self = re.search(r"You\s+are\s+Player\s+(\d+)", obs, re.I)
        if m_self:
            self_id = int(m_self.group(1))
        mentioned_ids = set()
        # Explicit "Player <id>" mentions
        mentioned_ids.update(int(x) for x in re.findall(r"Player\s*(\d+)", obs, flags=re.I))
        # Bracketed variants like "[Player 2]"
        mentioned_ids.update(int(x) for x in re.findall(r"\[Player\s*(\d+)\]", obs, flags=re.I))
        # Decision/chat tokens such as "[2 cooperate]" or "[0 chat]"
        mentioned_ids.update(
            int(x)
            for x in re.findall(r"\[\s*(\d+)\s+(?:cooperate|defect|chat)\b", obs, flags=re.I)
        )
        if not mentioned_ids:
            # Fallback: broader digit scrape while still rejecting non-player numbers
            mentioned_ids.update(int(x) for x in re.findall(r"\b(\d+)\b", obs))

        ids = sorted(i for i in mentioned_ids if 0 <= i <= 9)
        if ids:
            if self_id is not None:
                ctx["opponents"] = [i for i in ids if i != self_id]
            else:
                # Heuristic: in 3p-IPD there are usually 3 players -> take 2 smallest as opponents if unsure
                ctx["opponents"] = ids[:2]
        return ctx

    def _infer_ctx_truth_and_deception(self, obs: str) -> Dict:
        """Infer whether the current turn is chat or guess using instruction extraction.
        Falls back to last [GAME] line check if needed.
        """
        ctx: Dict = {}
        instr = extract_truth_and_deception_instruction(obs)
        if instr:
            ctx["instruction"] = instr
            if re.search(r"Now\s+guess\s+which\s+of\s+the\s+two\s+facts\s+are\s+correct", instr, re.I):
                ctx["phase"] = "guess"
                return ctx
        # Fallback: Look at the last [GAME] line
        game_lines = re.findall(r"^\s*\[GAME\]\s*(.*)$", obs or "", flags=re.I | re.M)
        last = game_lines[-1] if game_lines else ""
        if re.search(r"Now\s+guess\s+which\s+of\s+the\s+two\s+facts\s+are\s+correct", last, re.I):
            ctx["phase"] = "guess"
        else:
            ctx["phase"] = "chat"
        return ctx


__all__ = [
    "ActionNormalizer",
    "extract_last_game_line",
    "extract_truth_and_deception_instruction",
]
