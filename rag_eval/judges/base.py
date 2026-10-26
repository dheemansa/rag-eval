"""Abstract base class and null implementation for judges."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_eval.schema import JudgeVerdict


class BaseJudge(ABC):
    """Base class for all judges."""

    @abstractmethod
    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        """Judge whether a claim is entailed by the given context."""
        ...

    @abstractmethod
    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        """Judge whether a predicted answer is correct given the reference."""
        ...

    def judge_claims_batch(
        self, claims: list[str], context: str
    ) -> list[JudgeVerdict]:
        """Judge multiple claims against the same context. Default: loop."""
        return [self.judge_claim(c, context) for c in claims]


class NullJudge(BaseJudge):
    """A no-op judge that returns ``not_evaluated`` for every input.

    Used when the evaluation is run with ``--judge none``.
    """

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        return JudgeVerdict(
            claim=claim,
            label="not_evaluated",
            score=0.0,
            reason="Judge disabled (NullJudge).",
        )

    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        return JudgeVerdict(
            claim=predicted,
            label="not_evaluated",
            score=0.0,
            reason="Judge disabled (NullJudge).",
        )
