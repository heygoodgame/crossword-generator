"""Jeff Chen scored word list loader and lookup."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class DictionaryError(Exception):
    """Raised when the dictionary cannot be loaded or is empty after filtering."""


class Dictionary:
    """In-memory dictionary with score lookup and length-based indexing.

    Words are stored in uppercase internally.
    """

    def __init__(
        self,
        words: dict[str, int],
        *,
        min_word_score: int = 50,
        min_2letter_score: int = 30,
    ) -> None:
        self._words = words
        self._min_word_score = min_word_score
        self._min_2letter_score = min_2letter_score
        self._by_length: dict[int, list[str]] = defaultdict(list)
        for word in self._words:
            self._by_length[len(word)].append(word)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        min_word_score: int = 50,
        min_2letter_score: int = 30,
    ) -> Dictionary:
        """Load a dictionary from a word;score file.

        Args:
            path: Path to the dictionary file.
            min_word_score: Minimum score for words with 3+ letters.
            min_2letter_score: Minimum score for 2-letter words.

        Returns:
            A Dictionary instance with filtered words.

        Raises:
            DictionaryError: If the file is missing or empty after filtering.
        """
        path = Path(path)
        if not path.exists():
            raise DictionaryError(f"Dictionary file not found: {path}")

        words: dict[str, int] = {}
        for line_num, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 2:
                logger.warning("Malformed line %d: %r", line_num, line)
                continue
            word_raw, score_raw = parts
            word = word_raw.strip().upper()
            try:
                score = int(score_raw.strip())
            except ValueError:
                logger.warning("Invalid score on line %d: %r", line_num, line)
                continue

            if not word:
                logger.warning("Empty word on line %d", line_num)
                continue

            # Two-tier filtering
            if len(word) == 2:
                if score >= min_2letter_score:
                    words[word] = score
            else:
                if score >= min_word_score:
                    words[word] = score

        if not words:
            raise DictionaryError(
                f"Dictionary is empty after filtering "
                f"(min_word_score={min_word_score}, "
                f"min_2letter_score={min_2letter_score})"
            )

        return cls(
            words,
            min_word_score=min_word_score,
            min_2letter_score=min_2letter_score,
        )

    def contains(self, word: str) -> bool:
        """Check if a word is in the dictionary (case-insensitive)."""
        return word.upper() in self._words

    def score(self, word: str) -> int | None:
        """Return the score for a word, or None if not found."""
        return self._words.get(word.upper())

    def words_by_length(self, length: int) -> list[str]:
        """Return all words of the given length."""
        return self._by_length.get(length, [])

    def export_plain(self, output_path: Path | str, *, min_score: int = 50) -> int:
        """Write words to a plain text file (one lowercase word per line).

        Useful for creating a pre-filtered dictionary that external tools
        like go-crossword can ingest via their ``-dictionary`` flag.

        Args:
            output_path: Where to write the file.
            min_score: Minimum score threshold for included words.

        Returns:
            Number of words written.
        """
        output_path = Path(output_path)
        words = sorted(
            w.lower() for w, s in self._words.items() if s >= min_score
        )
        output_path.write_text("\n".join(words) + "\n" if words else "")
        logger.info(
            "Exported %d words (min_score=%d) to %s",
            len(words), min_score, output_path,
        )
        return len(words)

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def __len__(self) -> int:
        return len(self._words)
