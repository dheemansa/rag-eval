"""Judge backends for rag-eval.

A judge determines whether a claim is entailed by the provided context(s).
Three modes are supported:

* ``none``   – skip all judge-dependent metrics.
* ``replay`` – use cached verdicts; warn (or fail with ``--strict``) on miss.
* ``live``   – call the actual model (NLI or LLM API) and cache the result.
"""

from .api_judge import APIJudge
from .base import BaseJudge, NullJudge
from .cache import CachedJudge, CacheMissError, JudgeCache
from .nli import NLIJudge

__all__: list[str] = [
    "APIJudge",
    "BaseJudge",
    "CachedJudge",
    "CacheMissError",
    "JudgeCache",
    "NLIJudge",
    "NullJudge",
]
