"""Tests for the ClueGenerationStep pipeline step."""

from __future__ import annotations

import json
import re
import threading

import pytest

from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_generation import (
    build_clue_generation_prompt,
    build_clue_repair_prompt,
)
from crossword_generator.models import (
    ClueEntry,
    ClueGrade,
    FillResult,
    PuzzleDifficulty,
    PuzzleEnvelope,
    PuzzleType,
    ThemeConcept,
)
from crossword_generator.steps.clue_step import ClueGenerationStep

# A simple 5x5 grid with no black squares for testing
MOCK_GRID = [
    ["A", "B", "C", "D", "E"],
    ["F", "G", "H", "I", "J"],
    ["K", "L", "M", "N", "O"],
    ["P", "Q", "R", "S", "T"],
    ["U", "V", "W", "X", "Y"],
]

# Expected entries from compute_numbering for the mock grid
# 1-Across: ABCDE, 1-Down: AFKPU
# 2-Down: BGLQV, 3-Down: CHMRW, 4-Down: DINSX, 5-Down: EJOTY
# 6-Across: FGHIJ, 7-Across: KLMNO, 8-Across: PQRST, 9-Across: UVWXY


def _build_mock_clue_json(entries: list[NumberedEntry]) -> str:
    """Build a valid JSON response matching the expected entries."""
    clues = []
    for entry in entries:
        clues.append(
            {
                "number": entry.number,
                "direction": entry.direction,
                "clue": f"Clue for {entry.answer}",
            }
        )
    return json.dumps(clues)


class MockLLM(LLMProvider):
    """Mock LLM that returns a canned response."""

    def __init__(
        self,
        response: str | None = None,
        *,
        responses: list[str] | None = None,
    ) -> None:
        self._response = response
        self._responses = list(responses) if responses else []
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self._call_count += 1
        self.last_prompt = prompt
        self.last_system = kwargs.get("system")
        if self._responses:
            return self._responses.pop(0)
        return self._response or ""

    def is_available(self) -> bool:
        return True


def _make_envelope(
    *,
    grid: list[list[str]] | None = None,
    clues: list[ClueEntry] | None = None,
) -> PuzzleEnvelope:
    fill = None
    if grid is not None:
        fill = FillResult(grid=grid, filler_used="mock")
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=5,
        fill=fill,
        clues=clues or [],
    )


class _PromptAwareLLM(LLMProvider):
    """Mock LLM that answers exactly the entries named in each prompt.

    Parallel chunk generation can't use a fixed response queue (chunks finish
    in nondeterministic order), so this mock parses the ``N-DIRECTION: ANSWER``
    lines out of the user prompt and returns matching clue JSON. It records the
    order calls START in (under a lock) so a test can assert the first chunk
    ran before the fan-out, and can be told to raise on a specific answer to
    exercise failure propagation.
    """

    _LINE = re.compile(r"-\s*(\d+)-(ACROSS|DOWN):", re.IGNORECASE)

    def __init__(self, *, fail_on_answer: str | None = None) -> None:
        self.start_order: list[tuple[int, str]] = []
        self._fail_on_answer = fail_on_answer
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "prompt-aware-mock"

    def generate(self, prompt: str, **kwargs: object) -> str:
        pairs = [
            (int(n), d.lower()) for n, d in self._LINE.findall(prompt)
        ]
        with self._lock:
            self.start_order.append(pairs[0] if pairs else (-1, ""))
        if self._fail_on_answer and self._fail_on_answer in prompt:
            # Unparseable response => the chunk exhausts its retries and the
            # step raises, exercising failure propagation out of the fan-out.
            return "not valid json"
        clues = [
            {"number": n, "direction": d, "clue": f"Clue {n}{d}"}
            for n, d in pairs
        ]
        return json.dumps(clues)

    def is_available(self) -> bool:
        return True


class TestParallelChunkGeneration:
    """Warm-then-fan-out parallel chunk generation."""

    def _entries(self) -> list[NumberedEntry]:
        from crossword_generator.exporters.numbering import compute_numbering

        return compute_numbering(MOCK_GRID)

    def test_parallel_matches_serial_output(self) -> None:
        # Same chunking, same mock — parallel must produce identical clues in
        # identical order to serial, so manifests stay reproducible.
        serial = ClueGenerationStep(
            _PromptAwareLLM(), chunk_size=2, parallel_chunks=False
        ).run(_make_envelope(grid=MOCK_GRID))
        parallel = ClueGenerationStep(
            _PromptAwareLLM(),
            chunk_size=2,
            parallel_chunks=True,
            parallel_chunk_workers=4,
        ).run(_make_envelope(grid=MOCK_GRID))

        ser = [(c.number, c.direction, c.clue) for c in serial.clues]
        par = [(c.number, c.direction, c.clue) for c in parallel.clues]
        assert par == ser
        assert len(par) == len(self._entries())

    def test_first_chunk_warms_before_fanout(self) -> None:
        # The first chunk must START (and finish) before any other chunk
        # starts, so the rest read the warm cache instead of racing to create
        # it. With chunk_size=2 the first chunk's lead entry is (1, "across").
        llm = _PromptAwareLLM()
        ClueGenerationStep(
            llm,
            chunk_size=2,
            parallel_chunks=True,
            parallel_chunk_workers=4,
        ).run(_make_envelope(grid=MOCK_GRID))

        assert len(llm.start_order) > 1
        assert llm.start_order[0] == (1, "across")

    def test_parallel_falls_back_to_serial_for_single_chunk(self) -> None:
        # chunk_size >= entry count => one chunk => no fan-out, plain call.
        llm = _PromptAwareLLM()
        result = ClueGenerationStep(
            llm, chunk_size=999, parallel_chunks=True
        ).run(_make_envelope(grid=MOCK_GRID))
        assert len(result.clues) == len(self._entries())
        assert len(llm.start_order) == 1

    def test_failing_chunk_propagates(self) -> None:
        # A chunk that fails all retries must fail the whole puzzle, exactly as
        # in the serial path — the fan-out must not swallow it.
        llm = _PromptAwareLLM(fail_on_answer="UVWXY")
        step = ClueGenerationStep(
            llm,
            chunk_size=2,
            parallel_chunks=True,
            max_retries=1,
        )
        with pytest.raises(ValueError, match="Failed to parse clue response"):
            step.run(_make_envelope(grid=MOCK_GRID))


class TestClueGenerationStep:
    def test_happy_path(self) -> None:
        """Fill → clues populated correctly."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)
        for clue_entry in result.clues:
            assert clue_entry.clue.startswith("Clue for ")
            assert clue_entry.answer != ""

    def test_step_name(self) -> None:
        step = ClueGenerationStep(MockLLM())
        assert step.name == "clue-generation"

    def test_passes_cacheable_system_prompt(self) -> None:
        """Step splits prompt into static system + dynamic user."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        llm = MockLLM(response=mock_response)
        step = ClueGenerationStep(llm)
        envelope = _make_envelope(grid=MOCK_GRID)
        step.run(envelope)

        # Static rubric (role + guidelines + output format) goes to system
        assert llm.last_system is not None
        assert "expert crossword puzzle constructor" in llm.last_system
        assert "GUIDELINES:" in llm.last_system
        assert "OUTPUT FORMAT:" in llm.last_system

        # Per-puzzle entries go to the user prompt
        assert "ENTRIES TO CLUE:" in llm.last_prompt
        # And shouldn't be in system (else cache invalidates per puzzle)
        assert "ENTRIES TO CLUE:" not in llm.last_system

    def test_step_history_updated(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert "clue-generation" in result.step_history

    def test_validation_rejects_no_fill(self) -> None:
        step = ClueGenerationStep(MockLLM())
        envelope = _make_envelope(grid=None)
        with pytest.raises(ValueError, match="no fill result"):
            step.run(envelope)

    def test_chunked_generation_splits_into_multiple_calls(self) -> None:
        """chunk_size splits entries across calls; all clues are merged."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        chunk_size = 2
        # One response per chunk, each covering only that chunk's entries.
        chunks = [
            entries[i : i + chunk_size] for i in range(0, len(entries), chunk_size)
        ]
        responses = [_build_mock_clue_json(c) for c in chunks]

        llm = MockLLM(responses=responses)
        step = ClueGenerationStep(llm, chunk_size=chunk_size)
        result = step.run(_make_envelope(grid=MOCK_GRID))

        # Every entry got a clue, and the LLM was called once per chunk.
        assert len(result.clues) == len(entries)
        assert llm._call_count == len(chunks)

    def test_chunk_size_zero_is_single_call(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        llm = MockLLM(response=_build_mock_clue_json(entries))
        step = ClueGenerationStep(llm, chunk_size=0)
        result = step.run(_make_envelope(grid=MOCK_GRID))

        assert len(result.clues) == len(entries)
        assert llm._call_count == 1

    def test_validation_rejects_existing_clues(self) -> None:
        step = ClueGenerationStep(MockLLM())
        envelope = _make_envelope(
            grid=MOCK_GRID,
            clues=[
                ClueEntry(number=1, direction="across", answer="ABCDE", clue="test")
            ],
        )
        with pytest.raises(ValueError, match="already has clues"):
            step.run(envelope)

    def test_retry_on_malformed_response(self) -> None:
        """First response is garbage, second is valid JSON."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        good_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(
            MockLLM(responses=["not valid json!!!", good_response]),
            max_retries=3,
        )
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_all_retries_exhausted(self) -> None:
        """All retries return garbage → raises ValueError."""
        step = ClueGenerationStep(
            MockLLM(response="not json"),
            max_retries=2,
        )
        envelope = _make_envelope(grid=MOCK_GRID)
        with pytest.raises(ValueError, match="Failed to parse clue response"):
            step.run(envelope)

    def test_handles_markdown_wrapped_json(self) -> None:
        """LLM wraps JSON in ```json ... ``` fences."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        raw_json = _build_mock_clue_json(entries)
        wrapped = f"```json\n{raw_json}\n```"

        step = ClueGenerationStep(MockLLM(response=wrapped))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_autocorrects_wrong_direction(self) -> None:
        """LLM returns correct number but wrong direction for
        entries that only exist in one direction."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        # Build response with some directions flipped for
        # entries that only exist in one direction
        clues = []
        for entry in entries:
            d = entry.direction
            # Entries 6,7,8,9 across → flip to "down" (only
            # exist in one direction each, except 9 which has both)
            if entry.number in (6, 7, 8) and d == "across":
                d = "down"  # wrong, should be auto-corrected
            clues.append(
                {
                    "number": entry.number,
                    "direction": d,
                    "clue": f"Clue for {entry.answer}",
                }
            )
        mock_response = json.dumps(clues)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)
        # Verify the corrected entries have the right direction
        clue_map = {(c.number, c.direction): c for c in result.clues}
        assert (6, "across") in clue_map
        assert (7, "across") in clue_map
        assert (8, "across") in clue_map

        """LLM adds preamble text before the JSON array."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        raw_json = _build_mock_clue_json(entries)
        preamble = f"Here are the clues:\n\n{raw_json}"

        step = ClueGenerationStep(MockLLM(response=preamble))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_original_envelope_unchanged(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        step.run(envelope)

        assert envelope.clues == []  # Original unchanged


class TestComputeCrossingWords:
    def test_simple_grid(self) -> None:
        """Across entries cross all down entries in a full grid."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossings = compute_crossing_words(entries, MOCK_GRID)

        # 1-Across (ABCDE) should cross 1-Down (AFKPU), 2-Down (BGLQV), etc.
        across_1_crossings = crossings[(1, "across")]
        assert len(across_1_crossings) == 5  # Crosses all 5 down entries

        # 1-Down (AFKPU) should cross 1-Across (ABCDE), 6-Across (FGHIJ), etc.
        down_1_crossings = crossings[(1, "down")]
        assert len(down_1_crossings) == 5  # Crosses all 5 across entries

    def test_grid_with_black_squares(self) -> None:
        """Grid with black squares has fewer crossings."""
        grid = [
            ["A", "B", "."],
            ["C", "D", "E"],
            [".", "F", "G"],
        ]
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(grid)
        crossings = compute_crossing_words(entries, grid)

        # All entries should have crossings computed
        for entry in entries:
            key = (entry.number, entry.direction)
            assert key in crossings

    def test_returns_answer_words(self) -> None:
        """Crossing words should be the full answer words, not individual letters."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossings = compute_crossing_words(entries, MOCK_GRID)

        # All crossing words should be strings of length > 1
        for words in crossings.values():
            for word in words:
                assert len(word) > 1
                assert word.isalpha()


class TestThemeAnnotationsInPrompt:
    """Verify that theme entries are tagged in the clue generation prompt."""

    def test_theme_entries_tagged(self) -> None:
        """Seed entries in the grid get [THEME ENTRY] tags."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        # ABCDE is 1-Across, FGHIJ is 6-Across — use them as seeds
        theme = ThemeConcept(
            topic="Test theme",
            wordplay_type="hidden",
            revealer="KLMNO",  # 7-Across
            seed_entries=["ABCDE", "FGHIJ"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "ABCDE [THEME ENTRY]" in prompt
        assert "FGHIJ [THEME ENTRY]" in prompt

    def test_revealer_tagged(self) -> None:
        """The revealer entry gets a [REVEALER] tag."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Test theme",
            wordplay_type="hidden",
            revealer="KLMNO",  # 7-Across
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "KLMNO [REVEALER]" in prompt

    def test_revealer_position_in_guidance(self) -> None:
        """Theme block includes the revealer's number and direction."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Test theme",
            wordplay_type="hidden",
            revealer="KLMNO",  # 7-Across
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "7-Across" in prompt

    def test_no_theme_no_tags(self) -> None:
        """Without a theme, no entries are tagged."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(entries, crossing_words, PuzzleType.MINI)

        assert "[THEME ENTRY]" not in prompt
        assert "[REVEALER]" not in prompt

    def test_theme_prompt_includes_revealer_clue_as_guidance(self) -> None:
        """revealer_clue from theme is passed as context, not substituted."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Things that fly",
            wordplay_type="literal",
            revealer="KLMNO",
            seed_entries=["ABCDE"],
            revealer_clue="Soaring high above, or a hint to some other answers",
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "Revealer clue draft" in prompt
        assert "Soaring high above" in prompt

    def test_theme_prompt_variety_instructions(self) -> None:
        """Prompt instructs variety across theme entry clue styles."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Things that fly",
            wordplay_type="literal",
            revealer="KLMNO",
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "STANDALONE" in prompt
        assert "INDIRECT ALLUSION" in prompt
        assert "POSITIONAL CROSS-REFERENCE" in prompt
        assert "Vary the style" in prompt

    def test_theme_prompt_prohibits_bare_see_crossref(self) -> None:
        """Prompt warns against bare 'See X-Across' cross-references."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Test theme",
            wordplay_type="hidden",
            revealer="KLMNO",
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "See X-Across" in prompt
        assert "no solving context" in prompt

    def test_theme_prompt_revealer_accuracy_warning(self) -> None:
        """Prompt warns about connecting element vs full revealer word."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Test theme",
            wordplay_type="hidden",
            revealer="KLMNO",
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "connecting element" in prompt
        assert "component" in prompt
        assert "factually accurate" in prompt

    def test_theme_prompt_prohibits_one_of_revealer(self) -> None:
        """Prompt explicitly prohibits 'one of [REVEALER ANSWER]' formula."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)
        theme = ThemeConcept(
            topic="Things that fly",
            wordplay_type="literal",
            revealer="KLMNO",
            seed_entries=["ABCDE"],
        )

        prompt = build_clue_generation_prompt(
            entries, crossing_words, PuzzleType.MIDI, theme
        )

        assert "one of [REVEALER ANSWER]" in prompt
        assert "grammatically unnatural" in prompt

    def test_prompt_warns_against_sensitive_clue_wordplay(self) -> None:
        """Prompt asks for neutral treatment of sensitive answers."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(entries, crossing_words, PuzzleType.MINI)

        assert "underwear" in prompt
        assert "innuendo" in prompt
        assert "objectifying" in prompt

    def test_prompt_disallows_related_word_leakage(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
        )

        assert "related morphological variant/root" in prompt
        assert 'HOUSEWIFE with "Desperate ___wives"' in prompt
        assert "singular/plural forms" in prompt
        assert 'POL with "politician"' in prompt
        assert "abbreviation expansions" in prompt
        assert 'CEO with "executive"' in prompt

    def test_prompt_prioritizes_accuracy_and_exact_fit(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
            difficulty=PuzzleDifficulty.HARD,
        )

        assert "Accuracy is more important than cleverness" in prompt
        assert "Fill-in-the-blank clues must fit the answer exactly" in prompt
        assert "singular/plural" in prompt
        assert "GOT A SAY" in prompt

    def test_prompt_avoids_unpleasant_clue_wording(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
        )

        assert "death" in prompt
        assert "undocumented immigrant" in prompt
        assert "passed on" in prompt

    def test_easy_prompt_prioritizes_obvious_clues(self) -> None:
        """Easy guidance is easier than NYT Monday, not just mini/midi."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MIDI,
            difficulty=PuzzleDifficulty.EASY,
        )

        assert "easier than an NYT Monday" in prompt
        assert "totally obvious fill-in-the-blank" in prompt
        assert "choose instantly solvable" in prompt

    def test_hard_prompt_targets_tuesday_wordplay(self) -> None:
        """Hard guidance is tied to difficulty, even for mini puzzles."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
            difficulty=PuzzleDifficulty.HARD,
        )

        assert "solid NYT Tuesday" in prompt
        assert "Wednesday is the CEILING" in prompt
        assert "DO NOT STRAIN" in prompt
        assert "mild misdirection" in prompt
        assert "Saturday-level obscurity" in prompt
        assert "strained pop-culture references" in prompt
        assert "cross-generationally iconic" in prompt
        # Too-easy clues are an explicit defect on Hard puzzles.
        assert "TOO EASY" in prompt
        assert 'One more than two" for THREE' in prompt
        assert "DEFAULT to a harder fair angle" in prompt

    def test_prompt_applies_sliding_familiarity_to_references(self) -> None:
        """Older/nicher references must be especially well known."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MIDI,
            difficulty=PuzzleDifficulty.HARD,
        )

        assert "sliding familiarity standard" in prompt
        assert "the older or more niche the reference is" in prompt
        assert "one generation, fandom, or era" in prompt
        assert "everyday meaning" in prompt

    def test_prompt_disallows_word_count_tags_without_metadata(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
        )

        assert "Do not add word-count tags" in prompt
        assert "word-boundary metadata" in prompt

    def test_prompt_requires_parenthetical_explanatory_tags(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
        )

        assert "Put explanatory tags in parentheses" in prompt
        assert "To the ___ (in the extreme)" in prompt
        assert "Dennis ___ (pop art icon of soup cans)" in prompt

    def test_prompt_includes_prior_clues_to_avoid(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossing_words = compute_crossing_words(entries, MOCK_GRID)

        prompt = build_clue_generation_prompt(
            entries,
            crossing_words,
            PuzzleType.MINI,
            prior_clues_by_answer={"ABCDE": ["First five letters"]},
        )

        assert "PRIOR CLUES FOR THESE ANSWERS" in prompt
        assert "Do not repeat any of these exactly" in prompt
        assert "prefer a fresh" in prompt
        assert '- ABCDE: "First five letters"' in prompt

    def test_repair_prompt_applies_sliding_familiarity_to_references(self) -> None:
        """Repair should not replace bad references with another dated one."""
        bad_clue = ClueEntry(
            number=1,
            direction="across",
            answer="APPLE",
            clue="Singer Fiona",
        )
        grade = ClueGrade(
            number=1,
            direction="across",
            answer="APPLE",
            score=55,
            feedback="Too obscure for the audience.",
        )

        prompt = build_clue_repair_prompt(
            [(bad_clue, grade)],
            [bad_clue],
            {},
            PuzzleType.MINI,
            difficulty=PuzzleDifficulty.HARD,
        )

        assert "sliding familiarity standard" in prompt
        assert "the older or more niche the reference is" in prompt
        assert "plain accurate clue" in prompt


class TestRevealerClueNotSubstituted:
    """Verify the revealer clue from theme is NOT hard-substituted over LLM output."""

    def test_llm_revealer_clue_preserved(self) -> None:
        """The LLM's generated revealer clue is kept, not overwritten."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        theme = ThemeConcept(
            topic="Things that fly",
            wordplay_type="literal",
            revealer="KLMNO",  # 7-Across
            seed_entries=["ABCDE"],
            revealer_clue="Pre-generated revealer clue that should NOT appear",
        )

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=5,
            fill=FillResult(grid=MOCK_GRID, filler_used="mock"),
            theme=theme,
        )
        result = step.run(envelope)

        # Find the revealer's clue — should be the LLM-generated one
        revealer_clue = next(c for c in result.clues if c.answer == "KLMNO")
        assert revealer_clue.clue == "Clue for KLMNO"
        assert revealer_clue.clue != theme.revealer_clue
