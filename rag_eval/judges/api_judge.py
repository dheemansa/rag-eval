"""LLM-API-based judge for answer correctness and claim entailment."""

from __future__ import annotations

import json
import logging

from rag_eval.schema import JudgeVerdict

from .base import BaseJudge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLAIM_PROMPT = """\
You are an impartial NLI judge.  Determine whether the **claim** is entailed \
by the **context**.

### Context
{context}

### Claim
{claim}

Respond with ONLY a JSON object (no markdown fences):
{{"label": "<entailment|contradiction|neutral>", "confidence": <0.0-1.0>, "reason": "<brief explanation>"}}
"""

_ANSWER_PROMPT = """\
You are an impartial correctness judge.  Determine whether the **predicted \
answer** is correct given the **reference answer** for the question.

### Question
{question}

### Reference Answer
{reference}

### Predicted Answer
{predicted}

Respond with ONLY a JSON object (no markdown fences):
{{"label": "<correct|incorrect>", "confidence": <0.0-1.0>, "reason": "<brief explanation>"}}
"""


class APIJudge(BaseJudge):
    """Judge that delegates to an LLM via the OpenAI-compatible chat API.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``).
    api_key:
        Bearer token for the API.
    api_base:
        Base URL of the chat completions endpoint.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
    ) -> None:
        try:
            import httpx as _httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "APIJudge requires httpx. Install it with: pip install httpx"
            ) from exc

        self._model = model
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")

    # -- helpers ------------------------------------------------------------

    def _chat(self, user_prompt: str) -> str:
        """Send a single-user-message chat completion request."""
        import httpx

        url = f"{self._api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = httpx.post(
            url, headers=headers, json=payload, timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Best-effort JSON extraction from an LLM response."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (optionally with language tag)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        return json.loads(text.strip())

    def _safe_call(self, prompt: str, claim_text: str) -> JudgeVerdict:
        """Call the API and parse the response, returning an error verdict on failure."""
        try:
            raw = self._chat(prompt)
            parsed = self._parse_json(raw)
            return JudgeVerdict(
                claim=claim_text,
                label=parsed.get("label", "error"),
                score=float(parsed.get("confidence", 0.0)),
                reason=parsed.get("reason", ""),
            )
        except Exception:
            logger.exception("API judge call failed for: %s", claim_text)
            return JudgeVerdict(
                claim=claim_text,
                label="error",
                score=0.0,
                reason="API call or response parsing failed.",
            )

    # -- BaseJudge implementation -------------------------------------------

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        prompt = _CLAIM_PROMPT.format(claim=claim, context=context)
        return self._safe_call(prompt, claim)

    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        prompt = _ANSWER_PROMPT.format(
            question=question, predicted=predicted, reference=reference
        )
        return self._safe_call(prompt, predicted)
