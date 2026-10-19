"""Experiment adapter replaying the production rubric on individual clues.

This is the raw LLM grading pass, not a full puzzle/repair pipeline replay.
The binary output reflects fairness < 15, which can have non-content causes.
"""

from crossword_generator.classification_eval.providers import (
    GenerativeAdapter,
    InvalidResponseError,
)
from crossword_generator.graders.clue_grader import _parse_evaluation_response
from crossword_generator.llm.prompts.clue_evaluation import (
    build_clue_evaluation_messages,
)
from crossword_generator.models import ClueEntry, PuzzleType


class GraderAdapter:
    def __init__(self, config):
        self.provider = GenerativeAdapter(config | {"provider": "claude"}).provider

    def evaluate(self, state, questions):
        clue = ClueEntry(number=1, direction="across", **state)
        system, prompt = build_clue_evaluation_messages([clue], {}, PuzzleType.MINI)
        result = self.provider.generate_with_details(
            prompt, system=system, temperature=0
        )
        response = {
            "raw": result.response or {"text": result.text},
            "resolved_model": result.model,
            "usage": result.usage,
            "cost": result.cost,
        }
        try:
            grade = _parse_evaluation_response(result.text, [clue])[0]
            if (grade.number, grade.direction) != (1, "across"):
                raise ValueError("Wrong clue returned")
        except (ValueError, KeyError, TypeError):
            raise InvalidResponseError(response) from None
        response.update(
            grade=grade.model_dump(),
            probabilities={"fairness_repair": float(grade.fairness < 15)},
        )
        return response


def factory(config):
    return GraderAdapter(config)
