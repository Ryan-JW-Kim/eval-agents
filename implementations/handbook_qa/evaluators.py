"""Item-level evaluators for the grounded-QA handbook agent.

These graders score each agent answer on the three axes defined by
``evals/ground-truth/schema.json``:

1. **answer_correctness** - fuzzy/semantic similarity of the answer text to the
   reference answer (and acceptable variants), gated by ``must_include`` /
   ``must_not_include`` hard constraints.
2. **safety_level_match** - exact match of the agent's classified safety level
   against the expected level.
3. **traceability** - partial-credit recall of the *required* grounding sources
   the agent cited.

All three are plain Langfuse evaluator callables: pass them straight to
``run_experiment(..., evaluators=[...])``.

Agent output contract
---------------------
Evaluators accept the task ``output`` as either:

- a ``dict`` (preferred) shaped like::

    {
        "text": "<answer text>",
        "safety_level": "internal",          # optional
        "sources": [                          # optional, list of cited sources
            {"document_name": ..., "document_id": ..., "page": 12,
             "section_heading": ..., "chunk_id": ...},
            ...
        ],
    }

- or a plain ``str`` answer (older/simple tasks). In that case the safety and
  traceability evaluators degrade gracefully and report a zero score with an
  explanatory comment instead of failing the run.

The expected ground truth is read from ``expected_output`` (the reference answer
text) and ``metadata`` (safety level, traceability sources, answer-match config,
acceptable variants, must/must-not include), exactly as produced by
``data/langfuse_upload.py``.
"""

import logging
import re
import logging
import math
import re
import time
from difflib import SequenceMatcher
from typing import Any

from aieng.agent_evals.configs import Configs
from langfuse.experiment import Evaluation


logger = logging.getLogger(__name__)


DEFAULT_SIMILARITY_THRESHOLD = 0.8

#: Gemini embedding model used for ``embedding_cosine`` answer matching.
_EMBED_MODEL = "gemini-embedding-001"

#: Retry policy for transient embedding errors (e.g. rate limits under concurrency).
_EMBED_MAX_ATTEMPTS = 4
_EMBED_BACKOFF_SECONDS = 1.0

_embed_client: Any = None
_embed_cache: dict[str, list[float]] = {}
#: Set only for unrecoverable errors (bad key / model not found) so we stop
#: retrying; transient failures fall back to difflib per-call without disabling.
_embed_disabled = False


def _normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for stable comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _answer_text(output: Any) -> str:
    """Extract the answer string from a structured or plain-string output."""
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(output)


def _similarity(candidate: str, reference: str) -> float:
    """Return a normalized [0, 1] ``difflib`` similarity ratio over normalized text.

    Deterministic, dependency-free fallback used when embedding similarity is
    unavailable or not requested.
    """
    return SequenceMatcher(None, _normalize_text(candidate), _normalize_text(reference)).ratio()


def _is_unrecoverable_embed_error(message: str) -> bool:
    """True for errors that will not recover on retry (bad key / missing model)."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ("not_found", "404", "api key not valid", "api_key_invalid", "permission_denied", "401", "403")
    )


def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed texts with Gemini, caching by text. Returns ``None`` on failure.

    Uses the Gemini-enabled API key resolved by ``Configs.openai_api_key`` (its
    alias choices prefer ``OPENAI_API_KEY``). Transient errors (e.g. rate limits
    under concurrency) are retried with backoff and, if still failing, fall back
    to ``difflib`` for that call only. Unrecoverable errors (bad key / missing
    model) disable embeddings for the rest of the process.
    """
    global _embed_client, _embed_disabled
    if _embed_disabled:
        return None

    missing = [t for t in texts if t and t not in _embed_cache]
    if missing:
        try:
            from google import genai  # noqa: PLC0415

            if _embed_client is None:
                api_key = Configs().openai_api_key.get_secret_value()  # type: ignore[call-arg]
                _embed_client = genai.Client(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding client unavailable, using difflib: %s", exc)
            _embed_disabled = True
            return None

        for attempt in range(_EMBED_MAX_ATTEMPTS):
            try:
                response = _embed_client.models.embed_content(model=_EMBED_MODEL, contents=missing)
                for text, embedding in zip(missing, response.embeddings):
                    _embed_cache[text] = list(embedding.values)
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if _is_unrecoverable_embed_error(message):
                    logger.warning("embedding disabled (unrecoverable), using difflib: %s", exc)
                    _embed_disabled = True
                    return None
                if attempt < _EMBED_MAX_ATTEMPTS - 1:
                    time.sleep(_EMBED_BACKOFF_SECONDS * (2**attempt))
                    continue
                logger.warning("embedding failed after %d attempts, using difflib: %s", _EMBED_MAX_ATTEMPTS, exc)
                return None

    return [_embed_cache.get(t, []) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _best_similarity(answer: str, references: list[str], method: str) -> float:
    """Best similarity of ``answer`` against any reference.

    Uses embedding cosine when ``method == 'embedding_cosine'`` (falling back to
    ``difflib`` if embeddings are unavailable); otherwise uses ``difflib``.
    """
    refs = [r for r in references if r]
    if not refs or not answer:
        return 0.0

    if method == "embedding_cosine":
        vectors = _embed([answer, *refs])
        if vectors is not None:
            answer_vec, ref_vecs = vectors[0], vectors[1:]
            return max((_cosine(answer_vec, rv) for rv in ref_vecs), default=0.0)

    return max((_similarity(answer, r) for r in refs), default=0.0)


def _contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check on normalized text."""
    return _normalize_text(needle) in _normalize_text(haystack)


def answer_correctness_evaluator(
    *,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Score answer correctness with similarity plus hard include/exclude gates.

    Parameters
    ----------
    output : Any
        The agent task output (dict with ``text`` or a plain string).
    expected_output : Any
        The canonical reference answer text.
    metadata : dict[str, Any] | None
        Item metadata holding ``acceptable_variants``, ``must_include``,
        ``must_not_include`` and the optional ``answer_match`` config.

    Returns
    -------
    list[Evaluation]
        ``answer_similarity`` (numeric, best-matching variant) and
        ``answer_correctness`` (boolean: passes similarity threshold and all
        hard gates).
    """
    metadata = metadata or {}
    answer = _answer_text(output)
    reference = str(expected_output or "")

    answer_match = metadata.get("answer_match") or {}
    threshold = float(answer_match.get("threshold", DEFAULT_SIMILARITY_THRESHOLD))
    method = str(answer_match.get("method", "embedding_cosine"))

    # Best similarity across the reference and any acceptable variants.
    references = [reference, *metadata.get("acceptable_variants", [])]
    best_similarity = _best_similarity(answer, references, method)

    # Hard gates: every must_include term must appear; no must_not_include term may.
    missing = [term for term in metadata.get("must_include", []) if not _contains(answer, term)]
    forbidden = [term for term in metadata.get("must_not_include", []) if _contains(answer, term)]

    passes = best_similarity >= threshold and not missing and not forbidden

    comment_parts = [f"similarity={best_similarity:.3f} (threshold={threshold:.2f})"]
    if missing:
        comment_parts.append(f"missing required: {missing}")
    if forbidden:
        comment_parts.append(f"contains forbidden: {forbidden}")

    return [
        Evaluation(
            name="answer_similarity",
            value=best_similarity,
            data_type="NUMERIC",
            comment=f"best {method} similarity over {len(references)} candidate(s)",
        ),
        Evaluation(
            name="answer_correctness",
            value=passes,
            data_type="BOOLEAN",
            comment="; ".join(comment_parts),
            metadata={"missing_required": missing, "contains_forbidden": forbidden},
        ),
    ]


def safety_level_evaluator(
    *,
    output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> Evaluation:
    """Exact-match the agent's classified safety level against the expected level.

    Parameters
    ----------
    output : Any
        The agent task output. A structured ``dict`` is expected to carry a
        ``safety_level`` key; a plain string cannot, and is reported as a miss.
    metadata : dict[str, Any] | None
        Item metadata holding the expected ``safety_level``.

    Returns
    -------
    Evaluation
        ``safety_level_match`` (boolean exact match).
    """
    metadata = metadata or {}
    expected_level = metadata.get("safety_level")

    actual_level = output.get("safety_level") if isinstance(output, dict) else None

    if actual_level is None:
        return Evaluation(
            name="safety_level_match",
            value=False,
            data_type="BOOLEAN",
            comment="agent did not emit a 'safety_level'; expected "
            f"'{expected_level}'",
        )

    is_match = _normalize_text(str(actual_level)) == _normalize_text(str(expected_level))
    return Evaluation(
        name="safety_level_match",
        value=is_match,
        data_type="BOOLEAN",
        comment=f"expected '{expected_level}', got '{actual_level}'",
    )


def _source_key(source: dict[str, Any]) -> tuple[str, str, int | None, str]:
    """Build a comparable key from a source's identifying fields.

    ``chunk_id`` is intentionally excluded from the key because it is optional in
    the schema; document name, page, and section heading are the stable identity.
    """
    page = source.get("page")
    return (
        _normalize_text(str(source.get("document_name", ""))),
        _normalize_text(str(source.get("document_id", ""))),
        int(page) if isinstance(page, int) else None,
        _normalize_text(str(source.get("section_heading", ""))),
    )


def traceability_evaluator(
    *,
    output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Partial-credit traceability scoring against the required grounding sources.

    Recall is the fraction of *required* expected sources that the agent cited;
    precision is the fraction of cited sources that match any expected source.
    A boolean ``traceability_complete`` flags whether every required source was
    cited.

    Parameters
    ----------
    output : Any
        The agent task output. A structured ``dict`` is expected to carry a
        ``sources`` list; a plain string cannot, and is reported as zero recall.
    metadata : dict[str, Any] | None
        Item metadata holding the expected ``traceability`` source list.

    Returns
    -------
    list[Evaluation]
        ``traceability_recall`` (numeric), ``traceability_precision`` (numeric),
        and ``traceability_complete`` (boolean).
    """
    metadata = metadata or {}
    expected_sources: list[dict[str, Any]] = metadata.get("traceability", []) or []
    required_sources = [s for s in expected_sources if s.get("required", True)]

    cited_sources = output.get("sources", []) if isinstance(output, dict) else []
    cited_keys = {_source_key(s) for s in cited_sources if isinstance(s, dict)}
    expected_keys = {_source_key(s) for s in expected_sources}
    required_keys = {_source_key(s) for s in required_sources}

    matched_required = required_keys & cited_keys
    recall = len(matched_required) / len(required_keys) if required_keys else 1.0

    matched_any = expected_keys & cited_keys
    precision = len(matched_any) / len(cited_keys) if cited_keys else 0.0

    complete = required_keys.issubset(cited_keys)

    comment = (
        f"required cited {len(matched_required)}/{len(required_keys)} "
        f"(recall={recall:.2f}), precision={precision:.2f}"
    )

    return [
        Evaluation(
            name="traceability_recall",
            value=recall,
            data_type="NUMERIC",
            comment=comment,
        ),
        Evaluation(
            name="traceability_precision",
            value=precision,
            data_type="NUMERIC",
            comment=f"{len(matched_any)} of {len(cited_keys)} cited source(s) matched",
        ),
        Evaluation(
            name="traceability_complete",
            value=complete,
            data_type="BOOLEAN",
            comment="all required sources cited" if complete else "missing required source(s)",
        ),
    ]


__all__ = [
    "answer_correctness_evaluator",
    "safety_level_evaluator",
    "traceability_evaluator",
]
