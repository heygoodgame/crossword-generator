"""Tests for the clue fact-risk checker."""

from __future__ import annotations

import json

from crossword_generator.graders.clue_fact_checker import (
    ClueFactChecker,
    _fact_risk_reason,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import ClueEntry, PuzzleEnvelope, PuzzleType


class MockLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.call_count += 1
        return self.response

    def is_available(self) -> bool:
        return True


def test_fact_risk_reason_flags_risky_trivia() -> None:
    assert _fact_risk_reason("Hit song by Taylor Swift")
    assert _fact_risk_reason("Capital of France")
    assert _fact_risk_reason("First U.S. president")
    assert _fact_risk_reason("Feline pet") is None


def test_fact_risk_reason_flags_current_state_claims() -> None:
    stale = "current-state claim that may be stale"
    assert _fact_risk_reason("Channel currently airing playoff games") == stale
    assert _fact_risk_reason("Network that airs NBA games") == stale
    assert _fact_risk_reason("Shapiro who hosts a radio show") == stale
    assert _fact_risk_reason("Reigning MVP of the league") == stale
    assert _fact_risk_reason("Newest iPhone maker") == stale
    assert _fact_risk_reason("Quarterback who plays for Kansas City") == stale
    # Current-state takes precedence over other risk categories so the
    # fact-check prompt sees the staleness reason.
    assert _fact_risk_reason("Actor who currently stars on Broadway") == stale
    # Past-tense / timeless phrasings are not current-state claims.
    assert _fact_risk_reason("Feline pet") is None
    assert _fact_risk_reason("Joint above the shin") is None


def test_fact_checker_checks_only_risky_clues() -> None:
    response = json.dumps(
        [
            {
                "number": 1,
                "direction": "across",
                "status": "uncertain",
                "reason": "Specific song claim is not reliable.",
            }
        ]
    )
    llm = MockLLM(response)
    checker = ClueFactChecker(llm)
    envelope = PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=3,
        clues=[
            ClueEntry(
                number=1,
                direction="across",
                answer="CAT",
                clue="Hit song by Taylor Swift",
            ),
            ClueEntry(
                number=2,
                direction="down",
                answer="DOG",
                clue="Common canine pet",
            ),
        ],
    )

    results = checker.check(envelope)

    assert llm.call_count == 1
    assert len(results) == 1
    assert results[0].needs_repair is True
    assert results[0].answer == "CAT"
    assert results[0].risk_reason == "title or pop-culture claim"


def test_fact_checker_skips_when_no_risky_clues() -> None:
    llm = MockLLM("[]")
    checker = ClueFactChecker(llm)
    envelope = PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=3,
        clues=[
            ClueEntry(
                number=1,
                direction="across",
                answer="CAT",
                clue="Feline pet",
            ),
        ],
    )

    assert checker.check(envelope) == []
    assert llm.call_count == 0
