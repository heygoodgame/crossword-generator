"""Tests for hint generation: prompt parsing and the leak-screening step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.exporters.numbering import compute_numbering
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.hint_generation import (
    build_hint_generation_messages,
    parse_hint_response,
)
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
)
from crossword_generator.steps.hint_step import HintGenerationStep

# A tiny 3x3 with two across and two down entries.
GRID = [
    ["C", "A", "T"],
    ["O", ".", "O"],
    ["W", "O", "O"],
]


def _entries():
    return compute_numbering(GRID)


def _envelope() -> PuzzleEnvelope:
    env = PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        fill=FillResult(grid=GRID, filler_used="test"),
    )
    clues = [
        ClueEntry(number=n, direction=d, answer=a, clue=f"Clue for {a}")
        for (n, d, a) in [
            (e.number, e.direction, e.answer) for e in _entries()
        ]
    ]
    return env.model_copy(update={"clues": clues})


class StubLLM(LLMProvider):
    """Returns a scripted response per call; raises if it runs dry."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        self.calls += 1
        if not self._responses:
            raise AssertionError("StubLLM ran out of scripted responses")
        return self._responses.pop(0)

    def is_available(self) -> bool:
        return True


class TestParseHintResponse:
    def test_parses_with_preamble(self) -> None:
        entries = _entries()
        first = entries[0]
        raw = (
            "Here are the hints:\n"
            + json.dumps(
                [
                    {
                        "number": first.number,
                        "direction": first.direction,
                        "hint": "Easy hint",
                    }
                ]
            )
            + "\nDone."
        )
        hints = parse_hint_response(raw, entries)
        assert hints[(first.number, first.direction)] == "Easy hint"

    def test_autocorrects_unambiguous_direction(self) -> None:
        from collections import Counter

        entries = _entries()
        # Pick a number that exists in only one direction; if the model names
        # the wrong direction for it, the parser should fix it.
        counts = Counter(e.number for e in entries)
        solo = next(e for e in entries if counts[e.number] == 1)
        wrong = "down" if solo.direction == "across" else "across"
        raw = json.dumps(
            [{"number": solo.number, "direction": wrong, "hint": "X"}]
        )
        hints = parse_hint_response(raw, entries)
        assert hints[(solo.number, solo.direction)] == "X"

    def test_rejects_unknown_entry(self) -> None:
        with pytest.raises(ValueError):
            parse_hint_response(
                json.dumps(
                    [{"number": 99, "direction": "across", "hint": "X"}]
                ),
                _entries(),
            )

    def test_no_array_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_hint_response("no json here", _entries())


class TestHintGenerationStep:
    def _all_hints_response(self, hint: str = "Easy hint") -> str:
        return json.dumps(
            [
                {"number": e.number, "direction": e.direction, "hint": hint}
                for e in _entries()
            ]
        )

    def test_attaches_hints_to_clues(self) -> None:
        llm = StubLLM([self._all_hints_response("Frozen water")])
        step = HintGenerationStep(llm)
        out = step.run(_envelope())
        assert all(c.hint == "Frozen water" for c in out.clues)
        assert "hint-generation" in out.step_history

    def test_regenerates_leaking_hint(self) -> None:
        # First response leaks CAT (hint literally contains "cat"); second is
        # clean. The step should retry and end with the clean hint.
        entries = _entries()
        cat = next(e for e in entries if e.answer == "CAT")
        leaky = json.dumps(
            [
                {
                    "number": e.number,
                    "direction": e.direction,
                    "hint": "A cat says meow"
                    if e is cat
                    else "Clean hint",
                }
                for e in entries
            ]
        )
        clean_retry = json.dumps(
            [
                {
                    "number": cat.number,
                    "direction": cat.direction,
                    "hint": "Pet that purrs",
                }
            ]
        )
        llm = StubLLM([leaky, clean_retry])
        step = HintGenerationStep(llm, max_retries=3)
        out = step.run(_envelope())
        cat_clue = next(c for c in out.clues if c.answer == "CAT")
        assert cat_clue.hint == "Pet that purrs"
        assert llm.calls == 2

    def test_requires_clues(self) -> None:
        env = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI,
            fill=FillResult(grid=GRID, filler_used="test"),
        )
        step = HintGenerationStep(StubLLM([]))
        with pytest.raises(ValueError, match="no clues"):
            step.run(env)


def test_messages_include_answer_and_clue() -> None:
    entries = _entries()
    clues_by_key = {(e.number, e.direction): f"Real clue {e.answer}" for e in entries}
    system, user = build_hint_generation_messages(entries, clues_by_key)
    assert "HINT" in system
    assert "Real clue" in user
    # The first entry's answer should appear in the user block.
    assert entries[0].answer in user
