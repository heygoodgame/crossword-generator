"""Composite clue generation + quality grading pipeline step."""

from __future__ import annotations

import json
import logging

from crossword_generator.clue_history import (
    ClueHistoryIndex,
    DuplicateClueHit,
    duplicate_error_message,
)
from crossword_generator.exporters.numbering import (
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.graders.clue_fact_checker import ClueFactChecker
from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.graders.leak_detector import LeakFinding, detect_leaks
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_generation import (
    build_clue_repair_messages,
)
from crossword_generator.models import (
    ClueEntry,
    ClueGrade,
    ClueGradeReport,
    PuzzleEnvelope,
)
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.clue_step import ClueGenerationStep

logger = logging.getLogger(__name__)


class ClueWithGradingStep(PipelineStep):
    """Composite step: generate clues, grade quality, regenerate if below threshold.

    Wraps ClueGenerationStep and ClueGrader, looping up to max_retries attempts.
    Keeps the best-scoring clue set across all attempts.
    After the main loop, runs a single repair pass for any clue with a
    dangerously low accuracy sub-score, low individual score, or explicit
    evaluator feedback that flags a major issue.
    """

    def __init__(
        self,
        llm: LLMProvider,
        grader: ClueGrader,
        *,
        max_retries: int = 3,
        regenerate_on_fail: bool = True,
        accuracy_repair_threshold: float = 12,
        fairness_repair_threshold: float = 15,
        craft_repair_threshold: float = 8,
        freshness_repair_threshold: float = 0,
        individual_repair_score_threshold: float = 65,
        surgical_repair_pass_ratio: float | None = 0.8,
        fact_checker: ClueFactChecker | None = None,
        clue_history: ClueHistoryIndex | None = None,
        duplicate_repair_attempts: int = 4,
        leak_repair_attempts: int = 3,
        repair_verify_attempts: int = 2,
        generation_chunk_size: int = 0,
    ) -> None:
        self._llm = llm
        self._grader = grader
        self._max_retries = max_retries
        self._regenerate_on_fail = regenerate_on_fail
        self._accuracy_repair_threshold = accuracy_repair_threshold
        self._fairness_repair_threshold = fairness_repair_threshold
        self._craft_repair_threshold = craft_repair_threshold
        self._freshness_repair_threshold = freshness_repair_threshold
        self._individual_repair_score_threshold = individual_repair_score_threshold
        self._surgical_repair_pass_ratio = surgical_repair_pass_ratio
        self._fact_checker = fact_checker
        self._clue_history = clue_history
        self._duplicate_repair_attempts = duplicate_repair_attempts
        self._leak_repair_attempts = leak_repair_attempts
        self._repair_verify_attempts = repair_verify_attempts
        self._clue_step = ClueGenerationStep(
            llm, clue_history=clue_history, chunk_size=generation_chunk_size
        )

    @property
    def name(self) -> str:
        return "clue-generation-with-grading"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Generate and grade clues, retrying on low scores."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"ClueWithGradingStep validation failed: {'; '.join(errors)}"
            )

        max_attempts = self._max_retries if self._regenerate_on_fail else 1
        best_envelope: PuzzleEnvelope | None = None
        best_score: float = -1.0

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Clue generation+grading attempt %d/%d",
                attempt,
                max_attempts,
            )

            # Generate clues (clue_step expects empty clues)
            try:
                clued = self._clue_step.run(envelope)
            except ValueError:
                logger.warning("Attempt %d: clue generation failed", attempt)
                continue

            # Grade the clues
            report = self._grader.grade(clued)

            logger.info(
                "Attempt %d clue score: %.1f/100 (%s)",
                attempt,
                report.overall_score,
                "PASS" if report.passing else "FAIL",
            )

            # Populate per-clue quality scores from the grade report
            scored_clues = list(clued.clues)
            grade_lookup = {
                (g.number, g.direction): g.score for g in report.clue_grades
            }
            for i, clue in enumerate(scored_clues):
                key = (clue.number, clue.direction)
                if key in grade_lookup:
                    scored_clues[i] = clue.model_copy(
                        update={"quality_score": grade_lookup[key]}
                    )

            candidate = clued.model_copy(
                update={
                    "clues": scored_clues,
                    "clue_grade_report": report,
                }
            )

            if report.overall_score > best_score:
                best_score = report.overall_score
                best_envelope = candidate

            if report.passing:
                break

            # Mostly-good shortcut: if nearly all clues already pass, stop
            # regenerating the whole puzzle (which risks introducing new defects)
            # and let the surgical repair pass fix the few bad clues.
            if self._is_mostly_good(report):
                logger.info(
                    "Attempt %d mostly good (%d/%d clues clean); "
                    "skipping further regeneration in favor of surgical repair",
                    attempt,
                    self._clean_clue_count(report),
                    len(report.clue_grades),
                )
                break

        if best_envelope is None:
            raise ValueError(f"Clue generation failed on all {max_attempts} attempt(s)")

        # --- Surgical repair pass ---
        best_envelope = self._run_clue_repair(best_envelope)
        best_envelope = self._run_fact_check_repair(best_envelope)
        best_envelope = self._run_leak_repair(best_envelope)
        best_envelope = self._run_duplicate_clue_repairs(best_envelope)

        # Fix step_history: clue_step already added "clue-generation",
        # replace with our composite name
        step_history = [s for s in best_envelope.step_history if s != "clue-generation"]
        step_history.append(self.name)

        new_errors = list(best_envelope.errors)
        report = best_envelope.clue_grade_report
        if not report or not report.passing:
            new_errors.append(
                f"Clue quality below threshold after {max_attempts} attempt(s): "
                f"best score {best_score:.1f}"
            )

        # Soft-error surface for any leak that survived repair. The puzzle still
        # saves as a draft; the upload guard must refuse any "LEAK:" error.
        for finding in detect_leaks(best_envelope.clues):
            new_errors.append(_leak_error_message(finding))

        return best_envelope.model_copy(
            update={
                "step_history": step_history,
                "errors": new_errors,
            }
        )

    def _run_clue_repair(
        self,
        envelope: PuzzleEnvelope,
        forced_entries: list[tuple[ClueEntry, ClueGrade]] | None = None,
    ) -> PuzzleEnvelope:
        """Repair low-scoring or explicitly flawed clues, then verify.

        Runs an initial repair (including any ``forced_entries``), then re-checks
        the freshly re-graded clues and repairs any that are still flagged, up to
        ``repair_verify_attempts`` extra rounds. This catches clues a single
        repair pass fails to fix without re-running the whole pipeline.
        """
        current = self._run_clue_repair_once(envelope, forced_entries)
        # Only verify further if the first pass actually changed something —
        # a no-op (nothing flagged, or repair parse failure) returns the same
        # envelope object, and re-running would just repeat a failing call.
        if current is envelope:
            return current
        for _ in range(self._repair_verify_attempts):
            repaired = self._run_clue_repair_once(current)
            if repaired is current:
                break
            current = repaired
        return current

    def _run_clue_repair_once(
        self,
        envelope: PuzzleEnvelope,
        forced_entries: list[tuple[ClueEntry, ClueGrade]] | None = None,
    ) -> PuzzleEnvelope:
        """Single repair pass for low-scoring or explicitly flawed clues."""
        report = envelope.clue_grade_report
        forced_entries = forced_entries or []
        if not forced_entries and (not report or not report.clue_grades):
            return envelope

        # Find clues with accuracy below threshold
        grade_lookup: dict[tuple[int, str], ClueGrade] = {}
        if report:
            grade_lookup = {(g.number, g.direction): g for g in report.clue_grades}
        entries_to_repair: list[tuple[ClueEntry, ClueGrade]] = []
        repair_keys: set[tuple[int, str]] = set()
        for clue in envelope.clues:
            key = (clue.number, clue.direction)
            grade = grade_lookup.get(key)
            if grade and self._should_repair_grade(grade):
                entries_to_repair.append((clue, grade))
                repair_keys.add(key)
        for clue, grade in forced_entries:
            key = (clue.number, clue.direction)
            if key not in repair_keys:
                entries_to_repair.append((clue, grade))
                repair_keys.add(key)

        if not entries_to_repair:
            return envelope

        return self._repair_entries(envelope, entries_to_repair)

    def _run_fact_check_repair(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Run a second-pass fact-risk audit and repair uncertain clues."""
        if self._fact_checker is None:
            return envelope

        fact_results = self._fact_checker.check(envelope)
        if not fact_results:
            return envelope

        metadata = dict(envelope.metadata)
        metadata["clue_fact_check"] = [
            {
                "number": result.number,
                "direction": result.direction,
                "answer": result.answer,
                "clue": result.clue,
                "status": result.status,
                "reason": result.reason,
                "risk_reason": result.risk_reason,
            }
            for result in fact_results
        ]
        envelope = envelope.model_copy(update={"metadata": metadata})

        clue_lookup = {(clue.number, clue.direction): clue for clue in envelope.clues}
        entries_to_repair: list[tuple[ClueEntry, ClueGrade]] = []
        for result in fact_results:
            if not result.needs_repair:
                continue
            clue = clue_lookup.get((result.number, result.direction))
            if clue is None:
                continue
            feedback = (
                f"Fact check {result.status}: {result.reason} "
                f"(risk: {result.risk_reason}). Rewrite as a plain, "
                "factually safe clue unless the factual angle is airtight."
            )
            entries_to_repair.append(
                (
                    clue,
                    ClueGrade(
                        number=clue.number,
                        direction=clue.direction,
                        answer=clue.answer,
                        score=0.0,
                        feedback=feedback,
                        accuracy=0.0 if result.status == "incorrect" else 8.0,
                        freshness=10.0,
                        craft=10.0,
                        fairness=10.0,
                    ),
                )
            )

        if not entries_to_repair:
            return envelope

        repair_answers = [c.answer for c, _ in entries_to_repair]
        logger.info(
            "Clue fact-check repair: %d clue(s): %s",
            len(entries_to_repair),
            ", ".join(repair_answers),
        )
        return self._repair_entries(envelope, entries_to_repair)

    def _run_leak_repair(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Repair clues that mechanically leak their own answer.

        Uses the deterministic leak detector (no LLM) to find leaks, repairs
        them via the existing per-clue repair path, and re-checks up to
        ``leak_repair_attempts`` times. Leaks surviving all attempts are left
        in place and surfaced as soft errors by the caller.
        """
        current = envelope
        clue_lookup = {(c.number, c.direction): c for c in current.clues}
        for attempt in range(1, self._leak_repair_attempts + 1):
            findings = detect_leaks(current.clues)
            if not findings:
                return current

            logger.info(
                "Leak repair attempt %d/%d: %d leak(s): %s",
                attempt,
                self._leak_repair_attempts,
                len(findings),
                ", ".join(f.answer for f in findings),
            )

            clue_lookup = {(c.number, c.direction): c for c in current.clues}
            entries_to_repair: list[tuple[ClueEntry, ClueGrade]] = []
            for finding in findings:
                clue = clue_lookup.get((finding.number, finding.direction))
                if clue is None:
                    continue
                entries_to_repair.append((clue, _leak_clue_grade(finding)))

            if not entries_to_repair:
                return current

            repaired = self._repair_entries(current, entries_to_repair)
            if repaired is current:
                # Repair parse failed; stop trying.
                break
            current = repaired

        return current

    def _repair_entries(
        self,
        envelope: PuzzleEnvelope,
        entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
    ) -> PuzzleEnvelope:
        repair_answers = [c.answer for c, _ in entries_to_repair]
        logger.info(
            "Clue repair: %d clue(s) below threshold: %s",
            len(entries_to_repair),
            ", ".join(repair_answers),
        )

        # Build repair prompt
        assert envelope.fill is not None
        grid = envelope.fill.grid
        entries = compute_numbering(grid)
        crossing_words = compute_crossing_words(entries, grid)

        system_text, user_text = build_clue_repair_messages(
            entries_to_repair=entries_to_repair,
            all_clues=envelope.clues,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
            difficulty=envelope.difficulty,
        )

        # Call LLM for repair
        try:
            raw_response = self._llm.generate(user_text, system=system_text)
            repaired_clues = _parse_repair_response(raw_response, entries_to_repair)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Clue repair parse failed, keeping original clues: %s",
                exc,
            )
            return envelope

        # Swap repaired clues into the clue list
        repair_lookup = {(c.number, c.direction): c for c in repaired_clues}
        updated_clues = []
        for clue in envelope.clues:
            key = (clue.number, clue.direction)
            if key in repair_lookup:
                updated_clues.append(repair_lookup[key])
            else:
                updated_clues.append(clue)

        repaired_envelope = envelope.model_copy(update={"clues": updated_clues})

        # Re-grade to get updated scores
        previous_score = (
            envelope.clue_grade_report.overall_score
            if envelope.clue_grade_report
            else 0.0
        )
        new_report = self._grader.grade(repaired_envelope)

        logger.info(
            "Post-repair score: %.1f/100 (was %.1f/100)",
            new_report.overall_score,
            previous_score,
        )

        # Update quality scores on clues
        scored_clues = list(repaired_envelope.clues)
        new_grade_lookup = {
            (g.number, g.direction): g.score for g in new_report.clue_grades
        }
        for i, clue in enumerate(scored_clues):
            key = (clue.number, clue.direction)
            if key in new_grade_lookup:
                scored_clues[i] = clue.model_copy(
                    update={"quality_score": new_grade_lookup[key]}
                )

        return repaired_envelope.model_copy(
            update={
                "clues": scored_clues,
                "clue_grade_report": new_report,
            }
        )

    def _run_duplicate_clue_repairs(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        if self._clue_history is None:
            return envelope
        duplicate_hits = self._clue_history.find_duplicates(envelope.clues)
        return self._repair_duplicates(
            envelope, duplicate_hits, recheck_all_clues=True
        )

    def repair_external_duplicates(
        self, envelope: PuzzleEnvelope, duplicate_hits: list[DuplicateClueHit]
    ) -> PuzzleEnvelope:
        """Repair duplicate clues detected outside this step's own check.

        Used by the post-batch duplicate sweep: parallel workers can't see
        each other's in-flight clues, so cross-puzzle collisions are only
        detectable after every puzzle completes. The caller supplies the
        hits; replacement clues are still re-checked against the shared
        history before being accepted.
        """
        if self._clue_history is None or not duplicate_hits:
            return envelope
        return self._repair_duplicates(
            envelope, duplicate_hits, recheck_all_clues=False
        )

    def _repair_duplicates(
        self,
        envelope: PuzzleEnvelope,
        duplicate_hits: list[DuplicateClueHit],
        *,
        recheck_all_clues: bool,
    ) -> PuzzleEnvelope:
        """Informed-retry loop for duplicate clues, then soft-error fallback.

        With ``recheck_all_clues`` every clue is re-checked against the
        history after each repair pass (the in-run path). Without it, only
        the entries just repaired are re-checked — the external-sweep path,
        where the puzzle's own clues already live in the shared history and
        a full scan would flag every one of them as a self-match.
        """
        current = envelope
        for attempt in range(1, self._duplicate_repair_attempts + 1):
            if not duplicate_hits:
                return current
            logger.info(
                "Duplicate clue repair attempt %d/%d: %d exact duplicate(s)",
                attempt,
                self._duplicate_repair_attempts,
                len(duplicate_hits),
            )
            # Feed the FULL set of clues already used for each answer into the
            # repair prompt, not just the one we collided with — otherwise the
            # model can blindly land on another already-used clue.
            forced_entries = [
                (
                    hit.clue,
                    _duplicate_clue_grade(
                        hit,
                        self._clue_history.clues_for_answer(hit.clue.answer)
                        or [hit.existing_clue],
                    ),
                )
                for hit in duplicate_hits
            ]
            # This method has its own attempt loop + duplicate re-check, so use
            # the single-pass repair to avoid a nested verify loop.
            repaired = self._run_clue_repair_once(
                current,
                forced_entries=forced_entries,
            )
            if repaired is current:
                break
            current = repaired
            if recheck_all_clues:
                recheck_clues = current.clues
            else:
                repaired_keys = {
                    (hit.clue.number, hit.clue.direction)
                    for hit in duplicate_hits
                }
                recheck_clues = [
                    c
                    for c in current.clues
                    if (c.number, c.direction) in repaired_keys
                ]
            duplicate_hits = self._clue_history.find_duplicates(recheck_clues)

        # A duplicate that survives all repair attempts becomes a soft error —
        # the puzzle still completes and saves; the upload guard skips it. This
        # mirrors leak handling and avoids losing a whole puzzle over one
        # repeated clue.
        if duplicate_hits:
            new_errors = list(current.errors)
            for hit in duplicate_hits:
                new_errors.append(duplicate_error_message(hit))
                logger.warning(
                    'Duplicate clue survived repair: %s = "%s" (already: "%s")',
                    hit.clue.answer,
                    hit.clue.clue,
                    hit.existing_clue,
                )
            current = current.model_copy(update={"errors": new_errors})
        return current

    def _clean_clue_count(self, report: ClueGradeReport) -> int:
        """Number of graded clues that would not be sent to surgical repair."""
        return sum(1 for g in report.clue_grades if not self._should_repair_grade(g))

    def _is_mostly_good(self, report: ClueGradeReport) -> bool:
        """True when enough clues already pass that surgical repair of the rest
        is preferable to another whole-puzzle regeneration."""
        if self._surgical_repair_pass_ratio is None:
            return False
        total = len(report.clue_grades)
        if total == 0:
            return False
        return self._clean_clue_count(report) / total >= (
            self._surgical_repair_pass_ratio
        )

    def _should_repair_grade(self, grade: ClueGrade) -> bool:
        # Structured per-dimension signals take precedence over keyword matching.
        if (
            grade.accuracy is not None
            and grade.accuracy < self._accuracy_repair_threshold
        ):
            return True
        if (
            grade.fairness is not None
            and grade.fairness < self._fairness_repair_threshold
        ):
            return True
        if grade.craft is not None and grade.craft < self._craft_repair_threshold:
            return True
        # Freshness threshold is 0 (disabled) by default; Hard configs raise it
        # so evaluator-flagged too-easy clues (freshness 0-9) get repaired.
        if (
            grade.freshness is not None
            and grade.freshness < self._freshness_repair_threshold
        ):
            return True
        if grade.score < self._individual_repair_score_threshold:
            return True

        # Backstop: free-text feedback markers for issues not captured by the
        # numeric sub-scores.
        feedback = grade.feedback.lower()
        issue_markers = (
            "major issue",
            "factual error",
            "factually wrong",
            "incorrect",
            "not accurate",
            "inaccurate",
            "wrong proper noun",
            "wrong song",
            "wrong quote",
            "singular/plural",
            "tense mismatch",
            "grammar mismatch",
            "part-of-speech",
            "part of speech",
            "exact answer",
            "exact phrase",
            "does not fit",
            "not the same phrase",
            "unpleasant wording",
            "contains the answer",
            "part of the answer",
            "leaks",
            "should not be",
            "too obscure",
            "too easy",
            "ultra-current slang",
            "forced",
            "strained",
        )
        return any(marker in feedback for marker in issue_markers)

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if envelope.clues:
            errors.append("Envelope already has clues")
        return errors


def _parse_repair_response(
    raw_response: str,
    entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
) -> list[ClueEntry]:
    """Parse the LLM's repair response into ClueEntry objects.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the response structure is unexpected.
    """
    text = raw_response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found in repair response", text, 0)
    text = text[start : end + 1]
    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    # Build lookup of expected entries
    expected = {(c.number, c.direction): c for c, _ in entries_to_repair}

    repaired: list[ClueEntry] = []
    for item in parsed:
        number = int(item["number"])
        direction = item["direction"].lower()
        key = (number, direction)
        if key not in expected:
            raise ValueError(
                f"Repair response contains unexpected entry {number}-{direction}"
            )
        original = expected[key]
        repaired.append(
            ClueEntry(
                number=number,
                direction=direction,
                answer=original.answer,
                clue=str(item["clue"]),
            )
        )

    if len(repaired) != len(entries_to_repair):
        raise ValueError(
            f"Expected {len(entries_to_repair)} repaired clues, got {len(repaired)}"
        )

    return repaired


def _leak_feedback(finding: LeakFinding) -> str:
    """Human-readable repair instruction for a leaked clue."""
    if finding.kind == "exact":
        why = f'the clue contains the answer word "{finding.detail}" verbatim'
    elif finding.kind == "abbrev_expansion":
        why = f'the clue spells out the abbreviation as "{finding.detail}"'
    elif finding.kind == "abbrev_expansion_word":
        why = (
            f'the clue word "{finding.detail}" is one of the words the '
            f"abbreviation {finding.answer} stands for"
        )
    elif finding.kind == "irregular":
        why = f'the clue word "{finding.detail}" is a form of the answer'
    elif finding.kind == "shared_prefix":
        why = (
            f'the clue word "{finding.detail}" shares a root, etymology, or '
            f"spelling with the answer"
        )
    else:  # shared_root
        why = f'the clue word "{finding.detail}" shares a root with the answer'
    return (
        f"FAIRNESS LEAK: {why}. Rewrite the clue so it does not use the answer "
        f"word, any of its word-parts, roots, morphological variants, or "
        f"abbreviation expansions. Use a plain definition if needed."
    )


def _leak_clue_grade(finding: LeakFinding) -> ClueGrade:
    return ClueGrade(
        number=finding.number,
        direction=finding.direction,
        answer=finding.answer,
        score=0,
        accuracy=25,
        freshness=10,
        craft=10,
        fairness=0,
        feedback=_leak_feedback(finding),
    )


def _leak_error_message(finding: LeakFinding) -> str:
    return (
        f"LEAK: {finding.answer} ({finding.number}-{finding.direction}) "
        f'[{finding.kind}] in clue "{finding.clue}" (offending: '
        f'"{finding.detail}")'
    )


def _duplicate_clue_grade(
    hit: DuplicateClueHit, used_clues: list[str] | None = None
) -> ClueGrade:
    # List every clue already used for this answer so the model knows the full
    # set to avoid, not just the one it collided with.
    used = used_clues or [hit.existing_clue]
    used_list = "; ".join(f'"{c}"' for c in used)
    return ClueGrade(
        number=hit.clue.number,
        direction=hit.clue.direction,
        answer=hit.clue.answer,
        score=0,
        accuracy=25,
        freshness=0,
        craft=0,
        fairness=20,
        feedback=(
            f"This answer ({hit.clue.answer}) has already been clued as: "
            f"{used_list}. Write a NEW clue that is different from ALL of "
            "these. Use a fresh angle — a different definition, wordplay, or "
            "fill-in-the-blank."
        ),
    )


