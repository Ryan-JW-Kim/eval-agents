"""Shared helpers for the eval modules.

This module centralizes the small utilities and domain logic that are needed by
more than one eval file (heuristic + the LLM-judge modules). Anything specific
to a single module (e.g. the fuzzy similarity backends used only by the
heuristic evals) stays in that module.

Because the eval file names start with a digit, they can't be imported with a
plain ``import``. Use :func:`load_sibling_module` to load them::

    from importlib import import_module  # won't work: names start with a digit
    helpers = load_sibling_module("00_helpers", "helpers")

Contents:
    * Re-exported data stub types (AgentOutput, GroundTruthItem, ...).
    * ``_record`` -- the common flat result-record schema builder.
    * Text utilities: ``_normalize``, ``_tokens``.
    * Safety-label logic: ``parse_safety_level``, ``_SEVERITY_ORDER``,
      ``_SEVERITY_RANK``.
    * Citation/behavior helpers: ``_source_key``, ``_is_answerable``.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import re
import sys
from types import ModuleType
from typing import Any


# --------------------------------------------------------------------------- #
# Module loading (file names start with a digit -> can't use plain import)
# --------------------------------------------------------------------------- #
def load_sibling_module(file_stem: str, register_as: str | None = None) -> ModuleType:
    """Load a sibling ``.py`` file by stem (e.g. ``"00_data_stub"``).

    ``register_as`` optionally registers the module under a clean name in
    ``sys.modules`` so that dataclass annotations can resolve it.
    """
    path = pathlib.Path(__file__).with_name(f"{file_stem}.py")
    name = register_as or file_stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the data stubs once and re-export the types other modules rely on.
data_stub = load_sibling_module("00_data_stub", "data_stub")

AgentOutput = data_stub.AgentOutput
GroundTruthItem = data_stub.GroundTruthItem
SafetyLevel = data_stub.SafetyLevel
ExpectedBehavior = data_stub.ExpectedBehavior
Source = data_stub.Source
AgentTrace = data_stub.AgentTrace
TraceStep = data_stub.TraceStep
ToolName = data_stub.ToolName


# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #
# Safety levels ordered least -> most severe, for the under-rating check.
_SEVERITY_ORDER = [
    SafetyLevel.NEGLIGIBLE,
    SafetyLevel.LOW,
    SafetyLevel.MODERATE,
    SafetyLevel.HIGH,
    SafetyLevel.CRITICAL,
]
_SEVERITY_RANK = {lvl: i for i, lvl in enumerate(_SEVERITY_ORDER)}

_WORD_RE = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
def _record(eval_name: str, item_id: str, value: float, passed: bool, **details: Any) -> dict[str, Any]:
    """Build one flat result record (ready for DataFrame.from_records)."""
    rec: dict[str, Any] = {
        "id": item_id,
        "eval": eval_name,
        "value": float(value),
        "passed": bool(passed),
    }
    rec.update(details)
    return rec


def eval_result(records: list[dict[str, Any]] | None, eval_name: str) -> dict[str, Any] | None:
    """Find a prior eval record by name (for cross-eval gating); None if absent."""
    for r in records or []:
        if r.get("eval") == eval_name:
            return r
    return None


def skipped_record(eval_name: str, item_id: str, reason: str) -> dict[str, Any]:
    """A worst-score record for an eval we deliberately skipped (gated out).

    Scored 0.0 / not-passed with ``skipped=True`` so the row still appears in the
    table (e.g. an incorrect answer is treated as worst-case efficiency).
    """
    return _record(eval_name, item_id, value=0.0, passed=False, skipped=True, reason=reason)


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _normalize(text: str, case_sensitive: bool = False) -> str:
    """Collapse whitespace, drop punctuation; lowercase unless case-sensitive."""
    text = text if case_sensitive else text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str, case_sensitive: bool = False) -> list[str]:
    src = text if case_sensitive else text.lower()
    return _WORD_RE.findall(src)


# --------------------------------------------------------------------------- #
# Safety-label logic
# --------------------------------------------------------------------------- #
def parse_safety_level(raw: str | None) -> SafetyLevel | None:
    """Coerce a raw label string to SafetyLevel, or None if unrecognized."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    try:
        return SafetyLevel(key)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Citation / behavior helpers
# --------------------------------------------------------------------------- #
def _source_key(s: Source) -> tuple[str, int, str]:
    """Authoritative citation key: (document_id, page, normalized heading)."""
    return (s.document_id, s.page, _normalize(s.section_heading))


def _is_answerable(gt: GroundTruthItem) -> bool:
    return gt.expected_behavior == ExpectedBehavior.ANSWER


# --------------------------------------------------------------------------- #
# LLM endpoint (generic stubs -- swap in the real provider later)
# --------------------------------------------------------------------------- #
# These two functions are the ONLY places the eval suite touches an external
# model provider. Everything else (judges, embedding evals) is provider-
# agnostic and calls through here. When the endpoint is chosen, implement the
# bodies (OpenAI / Anthropic / local server / ...) and nothing else changes.
def llm_complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_mode: bool = True,
) -> str:
    """Single chat-completion call; returns the raw assistant text.

    STUB: wire up the chosen chat model/provider here. Keep the signature stable
    so callers (the LLM judges) never need to change.
    """
    raise NotImplementedError(
        "llm_complete is a stub; wire up the chosen chat model/provider here"
    )


def llm_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts; returns one vector per input, order preserved.

    STUB: wire up the chosen embedding model/provider here.
    """
    raise NotImplementedError(
        "llm_embed is a stub; wire up the chosen embedding model/provider here"
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_agent_trace(run_id: str) -> AgentTrace:
    """Return the agent's execution trace for a run.

    Generic stand-in for the real trace source: later this loads and parses
    ``logs/<uuid>/trace.jsonl`` into an ``AgentTrace``. For now it serves the
    canned example traces so the trace evals can be developed and tested.
    Callers (the trace judges) never need to change when the real loader lands.
    """
    trace = data_stub.EXAMPLE_TRACES_BY_ID.get(run_id)
    if trace is None:
        raise KeyError(f"No trace available for run_id={run_id!r}")
    return trace
