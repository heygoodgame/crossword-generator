"""LLM-based clue fact-risk checker."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_fact_check import (
    build_clue_fact_check_messages,
)
from crossword_generator.models import ClueEntry, PuzzleEnvelope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClueFactCheckResult:
    """Fact-check result for a single clue."""

    number: int
    direction: str
    answer: str
    clue: str
    status: str
    reason: str
    risk_reason: str

    @property
    def needs_repair(self) -> bool:
        return self.status in {"uncertain", "incorrect"}


class ClueFactChecker:
    """Checks risky clue/answer pairs for factual uncertainty.

    A deterministic pre-screen keeps the LLM call focused on clue forms that
    tend to hallucinate: proper-noun trivia, exact titles/quotes, dates,
    superlatives, and fill-in-the-blank phrase matches.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        enabled: bool = True,
        scope: str = "risky",
        max_parse_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._enabled = enabled
        self._scope = scope
        self._max_parse_retries = max_parse_retries

    def check(self, envelope: PuzzleEnvelope) -> list[ClueFactCheckResult]:
        """Fact-check risky clues and return any results."""
        if not self._enabled or not envelope.clues:
            return []

        risky_clues = self._select_clues(envelope.clues)
        if not risky_clues:
            return []

        system_text, user_text = build_clue_fact_check_messages(
            risky_clues,
            puzzle_type=envelope.puzzle_type,
            difficulty=envelope.difficulty,
        )

        parsed: list[ClueFactCheckResult] | None = None
        last_error = ""
        for attempt in range(1, self._max_parse_retries + 1):
            logger.info(
                "Clue fact-check attempt %d/%d using %s for %d risky clue(s)",
                attempt,
                self._max_parse_retries,
                self._llm.name,
                len(risky_clues),
            )
            raw_response = self._llm.generate(user_text, system=system_text)
            try:
                parsed = _parse_fact_check_response(raw_response, risky_clues)
                break
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse fact-check response: %s",
                    attempt,
                    last_error,
                )

        if parsed is None:
            logger.warning(
                "Failed to parse clue fact-check after %d attempt(s): %s",
                self._max_parse_retries,
                last_error,
            )
            return []

        return parsed

    def _select_clues(
        self, clues: list[ClueEntry]
    ) -> list[tuple[ClueEntry, str]]:
        if self._scope == "all":
            return [(clue, "configured all-clue fact check") for clue in clues]
        if self._scope != "risky":
            logger.warning(
                "Unknown clue fact-check scope %r; using risky", self._scope
            )

        selected: list[tuple[ClueEntry, str]] = []
        for clue in clues:
            risk_reason = _fact_risk_reason(clue.clue)
            if risk_reason:
                selected.append((clue, risk_reason))
        return selected


_FILL_IN_BLANK_RE = re.compile(r"(?:_{2,}|\bblank\b)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_QUOTED_TEXT_RE = re.compile(r"[\"'“”‘’].{2,}[\"'“”‘’]")
_TITLE_MARKER_RE = re.compile(
    r"\b(?:song|album|film|movie|novel|book|play|poem|show|series|"
    r"episode|hit|single|quote|line|title|anthem)\b",
    re.IGNORECASE,
)
_TRIVIA_MARKER_RE = re.compile(
    r"\b(?:actor|actress|author|composer|director|singer|rapper|band|"
    r"artist|athlete|president|prime minister|king|queen|team|mascot|"
    r"capital|country|state|city|currency|language|brand|company|"
    r"invented|founded|created|wrote|written by|sang|sung by|played by|"
    r"starred|starred in|directed|voiced|portrayed|hosted|won|winner|"
    r"nobel|oscar|grammy|emmy|tony|nba|nfl|mlb|nhl|fifa|olympic|"
    r"greek|roman|mythology|chemical|element|planet|moon|constellation)\b",
    re.IGNORECASE,
)
_SUPERLATIVE_RE = re.compile(
    r"\b(?:first|last|only|largest|smallest|oldest|youngest|highest|"
    r"lowest|longest|shortest|most|least|best-known|famous)\b",
    re.IGNORECASE,
)
_EXACT_CLAIM_RE = re.compile(
    r"\b(?:official|real name|nickname|birthplace|born|died|released|"
    r"set in|based on|member of|part of|home of)\b",
    re.IGNORECASE,
)


def _fact_risk_reason(clue: str) -> str | None:
    if _FILL_IN_BLANK_RE.search(clue):
        return "fill-in-the-blank exact phrase"
    if _YEAR_RE.search(clue):
        return "date or year claim"
    if _QUOTED_TEXT_RE.search(clue):
        return "quoted title or phrase"
    if _TITLE_MARKER_RE.search(clue):
        return "title or pop-culture claim"
    if _TRIVIA_MARKER_RE.search(clue):
        return "proper-noun or trivia claim"
    if _SUPERLATIVE_RE.search(clue):
        return "superlative factual claim"
    if _EXACT_CLAIM_RE.search(clue):
        return "exact factual claim"
    return None


def _parse_fact_check_response(
    raw_response: str,
    risky_clues: list[tuple[ClueEntry, str]],
) -> list[ClueFactCheckResult]:
    text = raw_response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found in response", text, 0)
    text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    expected = {
        (clue.number, clue.direction): (clue, risk_reason)
        for clue, risk_reason in risky_clues
    }

    results: list[ClueFactCheckResult] = []
    for item in parsed:
        number = int(item["number"])
        direction = str(item["direction"]).lower()
        key = (number, direction)
        if key not in expected:
            raise ValueError(
                f"Fact-check response contains unexpected entry {number}-{direction}"
            )
        clue, risk_reason = expected[key]
        status = str(item.get("status", "uncertain")).lower()
        if status not in {"safe", "uncertain", "incorrect"}:
            status = "uncertain"
        results.append(
            ClueFactCheckResult(
                number=number,
                direction=direction,
                answer=clue.answer,
                clue=clue.clue,
                status=status,
                reason=str(item.get("reason", "")),
                risk_reason=risk_reason,
            )
        )

    if len(results) != len(risky_clues):
        raise ValueError(
            f"Expected {len(risky_clues)} fact-check results, got {len(results)}"
        )
    return results
