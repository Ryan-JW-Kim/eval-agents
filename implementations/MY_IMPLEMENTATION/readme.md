# Eval Agents — Implementation

Evaluation suite for an agentic document-QA system over a safety-critical
handbook (LPG / NFPA 58 operations & maintenance). Each evaluation compares a
single agent **run** (`AgentOutput`) against a curated **ground truth**
(`GroundTruthItem`) and emits flat, tabular records ready for pandas.

> See [`context.md`](./context.md) for the wider project goal (PDF
> preprocessing → intermediate representation → chunking → agentic QA). This
> folder covers the **evaluation** half of that project.

---

## Progress at a glance

| Module | Status | What it covers |
| --- | --- | --- |
| `00_data_stub.py` | ✅ Done | Dataclasses for the GT schema + `AgentOutput`, plus example GT/output pairs. |
| `00_helpers.py` | ✅ Done | Shared, cross-module utilities (record schema, text/normalization, safety-label logic, citation helpers, module loading). |
| `01_heuristic_evals.py` | ✅ Done | 11 rule-based / fuzzy evals. No third-party deps (stdlib only). |
| `02_llm_judge_output_evals.py` | ⬜ Planned | LLM-as-judge scoring of the final response (quality, relevance, `embedding_cosine`). |
| `03_llm_judge_trace_evals.py` | ⬜ Planned | LLM-as-judge scoring of the agent's reasoning trace. |

Every record shares a common core — `id`, `eval`, `value` (float in `[0, 1]`),
`passed` (bool) — plus eval-specific detail columns. Missing detail columns
simply become `NaN` in the DataFrame, which is fine for aggregation.

```python
import pandas as pd
records = run_all_heuristics(gt, out)
df = pd.DataFrame.from_records(records)
```

---

## Architecture

```
00_data_stub.py     data model + example pairs  (no eval logic)
        │
        ▼
00_helpers.py       shared helpers used by ALL eval modules
        │
        ├──────────────┬──────────────────────┐
        ▼              ▼                      ▼
01_heuristic     02_llm_judge_output    03_llm_judge_trace
  _evals.py         _evals.py  (todo)       _evals.py  (todo)
```

Because the eval file names start with a digit, they can't be imported with a
plain `import`. `00_helpers.py` exposes `load_sibling_module(stem, register_as)`
to load them, and re-exports the data-stub types so downstream modules import
from one place.

### Shared helpers (`00_helpers.py`)

| Helper | Purpose |
| --- | --- |
| `load_sibling_module` | Import a digit-prefixed sibling `.py` file via `importlib`. |
| `_record` | Build one flat result record (the common `id/eval/value/passed` schema). |
| `_normalize` | Lowercase, strip punctuation, collapse whitespace for fair text comparison. |
| `_tokens` | Split text into word tokens (`\w+`) for set-based similarity. |
| `parse_safety_level` | Coerce a raw agent label string into a `SafetyLevel` enum (or `None`). |
| `_SEVERITY_ORDER` / `_SEVERITY_RANK` | Order safety levels least→most severe for under-rating checks. |
| `_source_key` | Canonical citation identity: `(document_id, page, normalized heading)`. |
| `_is_answerable` | Whether a GT item expects an answer vs. a refusal. |

---

## Eval catalogue

All 11 heuristic evals below live in `01_heuristic_evals.py`. They fall into two
groups: **structural** (inspect the output alone) and **comparison**
(output vs. ground truth). `value` is always in `[0, 1]`.

### Structural evals (output only)

| Eval | `value` meaning | Purpose / why it matters |
| --- | --- | --- |
| `required_headers` | Fraction of required `##` sections present **and** non-trivial. | Enforces the mandated response format (`Answer` / `Safety` / `Sources`). A well-formed answer is parseable and auditable; missing or empty sections signal an incomplete response. |
| `safety_label_valid` | `1.0` if the raw safety label parses to a valid enum, else `0.0`. | The agent must emit a machine-readable safety severity. An absent or garbage label can't be acted on and breaks every downstream safety check. |

### Comparison evals (ground truth + output)

| Eval | `value` meaning | Purpose / why it matters |
| --- | --- | --- |
| `safety_label_match` | `1.0` on exact safety-level match, else `0.0`. | Strict correctness of the hazard rating — the agent should classify severity exactly as the reference. |
| `safety_underrated` | `1.0` unless the agent rates **lower** severity than truth. | The safety-critical direction. Under-rating a hazard is dangerous; over-rating is merely conservative and passes. This is the metric to watch for life-safety regressions. |
| `has_required_citations` | `1.0` if citation presence matches the expected behavior. | Answerable questions must cite ≥ the required sources; refusals must cite **nothing** (grounding a non-answer is wrong). Behavior-aware grounding check. |
| `answer_similarity` | Best fuzzy-text similarity vs. reference answer + acceptable variants. | Measures semantic/lexical closeness of the answer to a known-good reference without demanding an exact string. Tolerant of paraphrase and length. |
| `must_include` | Fraction of required terms present (substring). | Guarantees critical facts/keywords (e.g. "inspected", "fitness") actually appear — catches answers that are fluent but omit the load-bearing content. |
| `must_not_include` | `1.0` unless any forbidden term appears. | Guardrail against unsafe or disallowed phrasing (e.g. "fill without", "anyone"). Any hit fails. |
| `citation_count_match` | Graded closeness of citation count to GT source count. | Rewards citing about the right number of sources — flags both under-citing (unsupported) and over-citing (padding). |
| `citation_precision` | Of the agent's citations, fraction that are genuine GT sources. | Penalizes fabricated / irrelevant citations (false positives). High precision = the agent isn't hallucinating references. |
| `citation_coverage` | Of the **required** GT sources, fraction the agent cited (recall). | Ensures the agent grounds its answer in the sources that actually matter — the recall complement to precision. |

#### Citation identity & similarity backends

- Citations are compared by **`_source_key`** = `(document_id, page, normalized
  heading)`, so a right answer with the wrong page (see the `omh-purge-001`
  example) is correctly counted as a citation miss.
- `answer_similarity` supports pluggable methods via `--method`:
  `token_set_ratio` (default, Dice coefficient over token sets — order- and
  length-tolerant), `levenshtein`/`difflib` (sequence ratio), and
  `embedding_cosine` (intentionally **not** implemented here — deferred to the
  LLM-judge modules).

---

## Running

```bash
cd eval-agents/implementations/MY_IMPLEMENTATION

# Run every heuristic over the built-in example pairs.
python 01_heuristic_evals.py
```

With pandas installed you get a table; without it, a plain per-record dump. The
example pairs in `00_data_stub.py` are deliberately mixed:

- `omh-dot-cyl-001` — agent answers correctly (most evals pass).
- `omh-purge-001` — agent under-rates safety, drops a required term, and cites
  the wrong page (safety + citation evals fail).
- `omh-oos-001` — out-of-scope question the agent correctly refuses.

---

## Data model (`00_data_stub.py`)

| Type | Role |
| --- | --- |
| `SafetyLevel` | `negligible → low → moderate → high → critical` severity enum. |
| `QuestionCategory` | `factual`, `procedural`, `table`, `synthesis`, `out_of_scope`, `adversarial`. |
| `ExpectedBehavior` | `answer`, `refuse_out_of_scope`, `refuse_unsafe`. |
| `Source` | A traceable citation (`document`, `page`, `section_heading`, `required`). |
| `ExpectedAnswer` | Reference `text` + `acceptable_variants` + `must_include` / `must_not_include`. |
| `GroundTruthItem` | One curated question with its expected answer, safety level, and traceability. |
| `AgentOutput` | One agent run: `answer_text` (markdown), `safety_level_raw`, `citations`. |

---

## Next steps

- [ ] `02_llm_judge_output_evals.py` — LLM-as-judge on the final response
      (answer quality, relevance) + `embedding_cosine` similarity backend.
- [ ] `03_llm_judge_trace_evals.py` — LLM-as-judge on the agent reasoning trace.
- [ ] Aggregation/reporting layer to roll per-record results into per-run and
      per-suite scorecards.
