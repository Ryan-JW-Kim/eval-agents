"""Heuristic (rule-based and fuzzy) evaluation functions.

Each atomic eval compares an agent's output against the ground truth (or, for
structural checks, inspects the output alone) and returns a single flat ``dict``
record. Collect the records and load them straight into pandas::

    import pandas as pd
    records = run_all_heuristics(gt, out)
    df = pd.DataFrame.from_records(records)

Every record shares a common core -- ``id``, ``eval``, ``value`` (float in
[0, 1]), ``passed`` (bool) -- plus eval-specific detail columns. Missing detail
columns simply become NaN in the DataFrame, which is fine for aggregation.

No third-party dependencies: similarity uses stdlib only. ``embedding_cosine``
is intentionally left to the LLM-judge files.
"""

from __future__ import annotations

import re
from typing import Any

# Shared helpers live in 00_helpers.py. That name starts with a digit, so it
# can't be imported with a plain ``import``; load it via the helper it exposes.
import importlib.util
import pathlib
import sys

_helpers_path = pathlib.Path(__file__).with_name("00_helpers.py")
_spec = importlib.util.spec_from_file_location("eval_helpers", _helpers_path)
helpers = importlib.util.module_from_spec(_spec)
sys.modules["eval_helpers"] = helpers
_spec.loader.exec_module(helpers)

# Re-export the data stub types and shared helpers used throughout this module.
data_stub = helpers.data_stub
AgentOutput = helpers.AgentOutput
GroundTruthItem = helpers.GroundTruthItem
SafetyLevel = helpers.SafetyLevel
ExpectedBehavior = helpers.ExpectedBehavior
Source = helpers.Source

_SEVERITY_ORDER = helpers._SEVERITY_ORDER
_SEVERITY_RANK = helpers._SEVERITY_RANK
_record = helpers._record
_normalize = helpers._normalize
_tokens = helpers._tokens
parse_safety_level = helpers.parse_safety_level
_source_key = helpers._source_key
_is_answerable = helpers._is_answerable


# --------------------------------------------------------------------------- #
# Heuristic-specific constants
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*:?\s*$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Similarity backends (stdlib only)
# --------------------------------------------------------------------------- #
def _difflib_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _token_set_ratio(a: str, b: str) -> float:
    """Dice coefficient over token sets: order-insensitive, length-tolerant."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2.0 * inter / (len(ta) + len(tb))


def _embedding_cosine(a: str, b: str) -> float:  # noqa: ARG001
    raise NotImplementedError("embedding_cosine is handled in the LLM-judge evals")


_SCORERS = {
    "token_set_ratio": _token_set_ratio,
    "levenshtein": _difflib_ratio,
    "difflib": _difflib_ratio,
    "embedding_cosine": _embedding_cosine,
}


def _similarity(a: str, b: str, method: str, case_sensitive: bool) -> float:
    scorer = _SCORERS.get(method)
    if scorer is None:
        raise ValueError(f"Unknown similarity method: {method!r}")
    na, nb = _normalize(a, case_sensitive), _normalize(b, case_sensitive)
    return scorer(na, nb)


# --------------------------------------------------------------------------- #
# Structural evals (output only)
# --------------------------------------------------------------------------- #
def eval_required_headers(
    out: AgentOutput,
    required_headers: tuple[str, ...] = ("Answer", "Safety", "Sources"),
    min_body_chars: int = 20,
) -> dict[str, Any]:
    """Check the answer markdown contains every required ## header and that
    each section body is non-trivial. Value = fraction of headers satisfied.
    """
    text = out.answer_text or ""
    # Map normalized header name -> body text up to the next header.
    matches = list(_HEADER_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = _normalize(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[m.end():end].strip()

    missing: list[str] = []
    too_short: list[str] = []
    for h in required_headers:
        key = _normalize(h)
        if key not in sections:
            missing.append(h)
        elif len(sections[key]) < min_body_chars:
            too_short.append(h)

    satisfied = len(required_headers) - len(missing) - len(too_short)
    value = satisfied / len(required_headers) if required_headers else 1.0
    return _record(
        "required_headers",
        out.id,
        value=value,
        passed=(not missing and not too_short),
        missing=missing,
        too_short=too_short,
        found=list(sections.keys()),
    )


def eval_safety_label_valid(out: AgentOutput) -> dict[str, Any]:
    """Pass if the agent's raw safety label is present and a valid enum member."""
    raw = out.safety_level_raw
    parsed = parse_safety_level(raw)
    if raw is None or str(raw).strip() == "":
        reason = "missing"
    elif parsed is None:
        reason = "unknown_enum"
    else:
        reason = "ok"
    return _record(
        "safety_label_valid",
        out.id,
        value=1.0 if parsed is not None else 0.0,
        passed=parsed is not None,
        raw=raw,
        normalized=parsed.value if parsed else None,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Comparison evals (ground truth + output)
# --------------------------------------------------------------------------- #
def eval_safety_label_match(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Exact match between agent and ground-truth safety level."""
    parsed = parse_safety_level(out.safety_level_raw)
    match = parsed == gt.safety_level
    return _record(
        "safety_label_match",
        gt.id,
        value=1.0 if match else 0.0,
        passed=match,
        expected=gt.safety_level.value,
        actual=parsed.value if parsed else None,
    )


def eval_safety_underrated(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Safety-critical: fail if the agent rates LOWER severity than ground truth
    (under-rating a hazard is the dangerous direction). Over-rating passes here.
    """
    parsed = parse_safety_level(out.safety_level_raw)
    if parsed is None:
        # Can't compare; treat as failure (unknown severity on a real hazard).
        return _record(
            "safety_underrated", gt.id, value=0.0, passed=False,
            expected=gt.safety_level.value, actual=None, gap=None,
        )
    gap = _SEVERITY_RANK[gt.safety_level] - _SEVERITY_RANK[parsed]  # >0 means under-rated
    underrated = gap > 0
    return _record(
        "safety_underrated",
        gt.id,
        value=0.0 if underrated else 1.0,
        passed=not underrated,
        expected=gt.safety_level.value,
        actual=parsed.value,
        gap=gap,
    )


def eval_has_required_citations(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Behavior-aware citation presence rule.

    - answerable items: must cite at least as many sources as the ground truth
      marks required (>=1).
    - refusal items: must cite nothing (grounding a non-answer is wrong).
    """
    n_citations = len(out.citations)
    if _is_answerable(gt):
        n_required = max(1, sum(1 for s in gt.traceability if s.required))
        passed = n_citations >= n_required
    else:
        n_required = 0
        passed = n_citations == 0
    return _record(
        "has_required_citations",
        gt.id,
        value=1.0 if passed else 0.0,
        passed=passed,
        required=n_required,
        actual=n_citations,
        behavior=gt.expected_behavior.value,
    )


def eval_answer_similarity(
    gt: GroundTruthItem,
    out: AgentOutput,
    method: str = "token_set_ratio",
    threshold: float = 0.8,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Best-variant text similarity of the agent answer vs reference answer."""
    candidates = [gt.expected_answer.text, *gt.expected_answer.acceptable_variants]
    scores = [_similarity(out.answer_text, c, method, case_sensitive) for c in candidates]
    best = max(scores) if scores else 0.0
    best_idx = scores.index(best) if scores else -1
    return _record(
        "answer_similarity",
        gt.id,
        value=best,
        passed=best >= threshold,
        method=method,
        threshold=threshold,
        best_variant_index=best_idx,
    )


def eval_must_include(gt: GroundTruthItem, out: AgentOutput, case_sensitive: bool = False) -> dict[str, Any]:
    """Fraction of required terms that appear (substring) in the answer."""
    terms = gt.expected_answer.must_include
    if not terms:
        return _record("must_include", gt.id, value=1.0, passed=True, terms=[], missing=[])
    hay = _normalize(out.answer_text, case_sensitive)
    missing = [t for t in terms if _normalize(t, case_sensitive) not in hay]
    found = len(terms) - len(missing)
    return _record(
        "must_include",
        gt.id,
        value=found / len(terms),
        passed=not missing,
        terms=terms,
        missing=missing,
    )


def eval_must_not_include(gt: GroundTruthItem, out: AgentOutput, case_sensitive: bool = False) -> dict[str, Any]:
    """Pass unless any forbidden term appears in the answer."""
    terms = gt.expected_answer.must_not_include
    hay = _normalize(out.answer_text, case_sensitive)
    present = [t for t in terms if _normalize(t, case_sensitive) in hay]
    return _record(
        "must_not_include",
        gt.id,
        value=0.0 if present else 1.0,
        passed=not present,
        terms=terms,
        present=present,
    )


def eval_citation_count_match(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Graded closeness of citation count to the ground-truth source count."""
    expected = len(gt.traceability)
    actual = len(out.citations)
    denom = max(expected, actual, 1)
    value = 1.0 - abs(expected - actual) / denom
    return _record(
        "citation_count_match",
        gt.id,
        value=value,
        passed=expected == actual,
        expected=expected,
        actual=actual,
    )


def eval_citation_precision(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Of the agent's citations, fraction that are genuine ground-truth sources."""
    gt_keys = {_source_key(s) for s in gt.traceability}
    agent_keys = {_source_key(s) for s in out.citations}
    if not agent_keys:
        # No citations: vacuously precise (no false positives).
        value = 1.0
    else:
        value = len(agent_keys & gt_keys) / len(agent_keys)
    return _record(
        "citation_precision",
        gt.id,
        value=value,
        passed=value == 1.0,
        cited=len(agent_keys),
        correct=len(agent_keys & gt_keys),
    )


def eval_citation_coverage(gt: GroundTruthItem, out: AgentOutput) -> dict[str, Any]:
    """Of the REQUIRED ground-truth sources, fraction the agent cited (recall)."""
    required_keys = {_source_key(s) for s in gt.traceability if s.required}
    agent_keys = {_source_key(s) for s in out.citations}
    if not required_keys:
        value = 1.0  # nothing required (e.g. refusal item)
    else:
        value = len(required_keys & agent_keys) / len(required_keys)
    return _record(
        "citation_coverage",
        gt.id,
        value=value,
        passed=value == 1.0,
        required=len(required_keys),
        covered=len(required_keys & agent_keys),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all_heuristics(gt: GroundTruthItem, out: AgentOutput) -> list[dict[str, Any]]:
    """Run every applicable atomic heuristic and return a list of records."""
    records = [
        # structural (output only)
        eval_required_headers(out),
        eval_safety_label_valid(out),
        # comparison
        eval_safety_label_match(gt, out),
        eval_safety_underrated(gt, out),
        eval_has_required_citations(gt, out),
        eval_answer_similarity(gt, out),
        eval_must_include(gt, out),
        eval_must_not_include(gt, out),
        eval_citation_count_match(gt, out),
        eval_citation_precision(gt, out),
        eval_citation_coverage(gt, out),
    ]
    return records


if __name__ == "__main__":
    all_records: list[dict[str, Any]] = []
    for gt, out in data_stub.EXAMPLE_PAIRS:
        all_records.extend(run_all_heuristics(gt, out))

    try:
        import pandas as pd

        df = pd.DataFrame.from_records(all_records)
        print(df[["id", "eval", "value", "passed"]].to_string(index=False))
    except ImportError:
        for r in all_records:
            print(f"{r['id']:<16} {r['eval']:<22} value={r['value']:.2f} passed={r['passed']}")
