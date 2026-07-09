"""Proper-noun classification for fill dictionary words.

Jeff's rule: a puzzle stops feeling like a word puzzle and starts feeling
like a trivia contest when too many answers are names. The fill grader
enforces a per-grid cap on proper-noun answers, but it is rule-based and
runs hundreds of times per puzzle, so the LLM classification happens
offline: the ``classify-proper-nouns`` CLI command labels every dictionary
word once and commits the result to a classification file. The grader
then just counts set membership.

Classification file format: ``WORD;P`` or ``WORD;C`` per line, uppercase,
sorted. ``P`` = the word is only viable in a crossword as a proper noun
(any reasonable clue must reference a specific named entity). ``C`` = the
word has at least one common-English reading a clue could use instead.
Words with both readings (AMBER, CHINA, MARK) are ``C`` — they only feel
like trivia when clued as names, which clue-side guidance handles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from crossword_generator.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROPER = "P"
COMMON = "C"

# How many times an unparsed/omitted word is re-queued before being left
# unclassified. Unclassified words are NOT counted as proper nouns, so the
# failure mode is a slightly lax cap, never a wrongly rejected fill.
MAX_CLASSIFY_RETRIES = 2

_SYSTEM_PROMPT = (
    "You classify crossword answers as proper-noun-only or not.\n"
    "\n"
    "Label a word P when EVERY reasonable crossword clue for it would have "
    "to reference a specific named entity: a person or personal name "
    "(AARON, OPRAH), place (ERIE, OHIO), brand or product (OREO), "
    "organization, team, or initialism of one (NBA, UCLA), title of a work "
    "(SHREK), or fictional character (ELSA).\n"
    "\n"
    "Label a word C when it has at least one ordinary common-English "
    "reading — a standard noun, verb, adjective, interjection, or phrase — "
    "that a clue could use without naming any specific entity. Words that "
    "are both a name and a common word are C: AMBER (fossil resin), CHINA "
    "(dishes), MARK (a spot), SUE (to litigate), DELTA (river mouth).\n"
    "\n"
    "Entries may be multi-word phrases run together (ICEDTEA = ICED TEA); "
    "judge the whole phrase. Obscure but real common words are still C.\n"
    "\n"
    "Respond with one line per input word, in order, formatted exactly as\n"
    "WORD P\n"
    "or\n"
    "WORD C\n"
    "with no other text, numbering, or commentary."
)


def build_classification_prompt(words: list[str]) -> str:
    """Build the user prompt for one classification batch."""
    return "Classify these crossword answers:\n\n" + "\n".join(words)


def parse_classification_response(
    raw: str, expected_words: list[str]
) -> dict[str, str]:
    """Parse ``WORD P|C`` lines, returning labels for expected words only.

    Words missing from the response or carrying an unknown label are simply
    absent from the result; the caller re-queues them.
    """
    expected = {w.upper() for w in expected_words}
    labels: dict[str, str] = {}
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        word, label = parts[0].upper(), parts[1].upper()
        if word in expected and label in (PROPER, COMMON):
            labels[word] = label
    return labels


def classify_words(
    provider: LLMProvider,
    words: list[str],
    *,
    batch_size: int = 100,
    max_workers: int = 4,
    model: str | None = None,
    checkpoint: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    """Classify words in parallel batches, retrying omitted words.

    Returns a mapping of uppercase word -> ``P``/``C``. Words the model
    repeatedly fails to label are omitted (and logged). ``checkpoint`` is
    called with the labels accumulated so far after each round, so a long
    run interrupted mid-way keeps its completed work.
    """
    pending = sorted({w.strip().upper() for w in words if w.strip()})
    results: dict[str, str] = {}

    for round_number in range(MAX_CLASSIFY_RETRIES + 1):
        if not pending:
            break
        batches = [
            pending[i : i + batch_size]
            for i in range(0, len(pending), batch_size)
        ]
        logger.info(
            "Classification round %d: %d words in %d batch(es)",
            round_number + 1,
            len(pending),
            len(batches),
        )

        def _classify_batch(batch: list[str]) -> dict[str, str]:
            kwargs: dict[str, object] = {"temperature": 0.0}
            if model:
                kwargs["model"] = model
            # A single failed batch (truncation, transient API error) must
            # not kill the run — its words simply requeue for the next round.
            try:
                raw = provider.generate(
                    build_classification_prompt(batch),
                    system=_SYSTEM_PROMPT,
                    **kwargs,
                )
            except Exception:
                logger.exception(
                    "Classification batch of %d words failed; requeueing",
                    len(batch),
                )
                return {}
            return parse_classification_response(raw, batch)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, labels in enumerate(
                executor.map(_classify_batch, batches), start=1
            ):
                results.update(labels)
                if checkpoint is not None and index % 20 == 0:
                    checkpoint(dict(results))
                    logger.info(
                        "Checkpoint: %d/%d batches, %d words labeled",
                        index,
                        len(batches),
                        len(results),
                    )

        if checkpoint is not None:
            checkpoint(dict(results))
        pending = [w for w in pending if w not in results]

    if pending:
        logger.warning(
            "%d word(s) left unclassified after %d round(s): %s",
            len(pending),
            MAX_CLASSIFY_RETRIES + 1,
            pending[:20],
        )
    return results


def load_classifications(path: Path) -> dict[str, str]:
    """Load an existing classification file (``WORD;P|C`` per line)."""
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        word, _, label = line.partition(";")
        word, label = word.strip().upper(), label.strip().upper()
        if word and label in (PROPER, COMMON):
            labels[word] = label
    return labels


def save_classifications(path: Path, labels: dict[str, str]) -> None:
    """Write the classification file sorted for stable diffs."""
    lines = [f"{word};{labels[word]}" for word in sorted(labels)]
    path.write_text("\n".join(lines) + "\n")


def load_proper_noun_set(path: Path) -> frozenset[str]:
    """Load only the proper-noun words for the fill grader."""
    if not path.exists():
        raise FileNotFoundError(
            f"Proper-noun classification file not found: {path}"
        )
    labels = load_classifications(path)
    return frozenset(
        word for word, label in labels.items() if label == PROPER
    )
