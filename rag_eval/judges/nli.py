"""NLI-based judge using a cross-encoder model."""

from __future__ import annotations

import logging
from typing import Any

from rag_eval.schema import JudgeVerdict

from .base import BaseJudge

logger = logging.getLogger(__name__)

# Label ordering expected by DeBERTa-v3 NLI models
_NLI_LABELS = ["contradiction", "entailment", "neutral"]


def _require_nli_deps() -> None:
    """Raise a friendly error if NLI dependencies are missing."""
    try:
        import torch as _torch  # noqa: F401
        import transformers as _transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NLI judge requires PyTorch and HuggingFace Transformers. "
            "Install them with:  pip install rag-eval[nli]"
        ) from exc


class NLIJudge(BaseJudge):
    """Judge that uses a cross-encoder NLI model for entailment checking.

    The model is lazily loaded on first use so that importing this module is
    cheap and does not require GPU availability.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for a cross-encoder NLI model.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).  If ``None``,
        CUDA is used when available.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str | None = None,
    ) -> None:
        _require_nli_deps()
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading NLI model %s on %s …", self._model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        ).to(device)
        self._model.eval()
        self._device = device

    # -- internal inference -------------------------------------------------

    def _predict(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[tuple[str, float]]:
        """Run the cross-encoder and return ``(label, probability)`` per pair."""
        import torch

        self._ensure_model()

        encoded = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)

        results: list[tuple[str, float]] = []
        for row in probs:
            idx = int(row.argmax())
            results.append((_NLI_LABELS[idx], float(row[idx])))
        return results

    # -- BaseJudge implementation -------------------------------------------

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        (label, score) = self._predict([context], [claim])[0]
        return JudgeVerdict(claim=claim, label=label, score=score)

    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        # Treat the reference answer as the premise and the predicted answer
        # as the hypothesis for entailment checking.
        (label, score) = self._predict([reference], [predicted])[0]
        return JudgeVerdict(claim=predicted, label=label, score=score)

    def judge_claims_batch(
        self, claims: list[str], context: str
    ) -> list[JudgeVerdict]:
        """Process all claims in a single batched forward pass."""
        if not claims:
            return []
        premises = [context] * len(claims)
        results = self._predict(premises, claims)
        return [
            JudgeVerdict(claim=c, label=lbl, score=sc)
            for c, (lbl, sc) in zip(claims, results)
        ]
