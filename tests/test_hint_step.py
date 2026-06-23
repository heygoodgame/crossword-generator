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

    def test_repairs_leaking_hint(self) -> None:
        # First (generation) response leaks CAT (hint literally contains
        # "cat"); the convergence loop should repair it via a second call and
        # end with the clean hint.
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
        repair = json.dumps(
            [
                {
                    "number": cat.number,
                    "direction": cat.direction,
                    "hint": "Pet that purrs",
                }
            ]
        )
        llm = StubLLM([leaky, repair])
        step = HintGenerationStep(llm, max_retries=3, max_repair_rounds=3)
        out = step.run(_envelope())
        cat_clue = next(c for c in out.clues if c.answer == "CAT")
        assert cat_clue.hint == "Pet that purrs"
        assert llm.calls == 2

    def test_drops_persistently_leaking_hint(self) -> None:
        # A hint that keeps leaking through every repair round is dropped — the
        # entry ships with no hint rather than a leaking one.
        entries = _entries()
        cat = next(e for e in entries if e.answer == "CAT")

        def leaky_for_cat(only_cat: bool) -> str:
            src = [cat] if only_cat else entries
            return json.dumps(
                [
                    {
                        "number": e.number,
                        "direction": e.direction,
                        "hint": "A cat naps" if e.answer == "CAT" else "Clean",
                    }
                    for e in src
                ]
            )

        # generation + max_repair_rounds repair calls, all still leaking.
        responses = [leaky_for_cat(only_cat=False)] + [
            leaky_for_cat(only_cat=True) for _ in range(3)
        ]
        llm = StubLLM(responses)
        step = HintGenerationStep(llm, max_retries=3, max_repair_rounds=3)
        out = step.run(_envelope())
        cat_clue = next(c for c in out.clues if c.answer == "CAT")
        assert cat_clue.hint == ""
        # Other entries kept their clean hints.
        assert all(c.hint == "Clean" for c in out.clues if c.answer != "CAT")

    def test_optional_hint_skipped_when_omitted(self) -> None:
        # The model deliberately skips one entry (omits it / empty hint). That
        # entry ships hint-less, with no retry churn.
        entries = _entries()
        skip = entries[0]
        resp = json.dumps(
            [
                {"number": e.number, "direction": e.direction, "hint": "Easy"}
                for e in entries
                if e is not skip
            ]
        )
        llm = StubLLM([resp])
        step = HintGenerationStep(llm, max_retries=3)
        out = step.run(_envelope())
        skipped = next(
            c
            for c in out.clues
            if c.number == skip.number and c.direction == skip.direction
        )
        assert skipped.hint == ""
        # No wasted retries: a single generation call covered the chunk.
        assert llm.calls == 1

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


def test_generation_prompt_allows_optional_empty_hint() -> None:
    entries = _entries()
    clues_by_key = {(e.number, e.direction): "Clue" for e in entries}
    system, user = build_hint_generation_messages(entries, clues_by_key)
    # The prompt must tell the model it may skip with an empty hint.
    assert "OPTIONAL" in system
    assert '""' in system


def test_repair_messages_include_problem_and_current_hint() -> None:
    from crossword_generator.llm.prompts.hint_generation import (
        build_hint_repair_messages,
    )

    entries = _entries()[:1]
    e = entries[0]
    key = (e.number, e.direction)
    system, user = build_hint_repair_messages(
        entries,
        {key: "Real clue"},
        {key: "Broken hint that leaks"},
        {key: ["the hint leaks the answer (exact)"]},
    )
    assert "REPAIRING" in system
    assert "Broken hint that leaks" in user
    assert "leaks the answer" in user


class _FakeFactChecker:
    """Flags any hint whose text contains a marker phrase, as the real
    checker keys on hint content (so a repaired hint passes)."""

    def __init__(self, bad_phrase: str) -> None:
        self._bad_phrase = bad_phrase

    def check(self, envelope):  # type: ignore[no-untyped-def]
        from crossword_generator.graders.clue_fact_checker import (
            ClueFactCheckResult,
        )

        results = []
        for c in envelope.clues:
            # In the shadow envelope, c.clue holds the hint text.
            if self._bad_phrase in c.clue:
                results.append(
                    ClueFactCheckResult(
                        number=c.number,
                        direction=c.direction,
                        answer=c.answer,
                        clue=c.clue,
                        status="incorrect",
                        reason="planted false fact",
                        risk_reason="test",
                    )
                )
        return results


def test_fact_flagged_hint_is_repaired() -> None:
    entries = _entries()
    cat = next(e for e in entries if e.answer == "CAT")
    gen = json.dumps(
        [
            {
                "number": e.number,
                "direction": e.direction,
                "hint": "Discovered in 1999" if e.answer == "CAT" else "Clean",
            }
            for e in entries
        ]
    )
    repair = json.dumps(
        [{"number": cat.number, "direction": cat.direction, "hint": "Purring pet"}]
    )
    llm = StubLLM([gen, repair])
    step = HintGenerationStep(
        llm,
        max_retries=3,
        max_repair_rounds=3,
        fact_checker=_FakeFactChecker(bad_phrase="1999"),
    )
    out = step.run(_envelope())
    cat_clue = next(c for c in out.clues if c.answer == "CAT")
    assert cat_clue.hint == "Purring pet"
    assert llm.calls == 2
