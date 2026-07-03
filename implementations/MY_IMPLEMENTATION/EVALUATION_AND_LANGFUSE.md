# Evaluation and Langfuse Guide

This document explains how to run the PDF QA pipeline, how to regenerate and analyze evaluation output, and where to inspect the corresponding traces in Langfuse.

Relevant code:
- [main2.py](implementations/MY_IMPLEMENTATION/main2.py)
- [evaluate_main2.py](implementations/MY_IMPLEMENTATION/evaluate_main2.py)
- [analyze_main2_eval.py](implementations/MY_IMPLEMENTATION/analyze_main2_eval.py)
- [langfuse_tracing.py](implementations/MY_IMPLEMENTATION/langfuse_tracing.py)
- [main2_eval_results.json](implementations/MY_IMPLEMENTATION/main2_eval_results.json)

## Quick Start

Run one question through the pipeline:

```bash
python3 implementations/MY_IMPLEMENTATION/main2.py --query "What is the minimum required capacity and rating of the portable fire extinguisher each bulk plant must have?"
```

Run the evaluation set and overwrite the checked-in results file:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --max-items 100 --output-path implementations/MY_IMPLEMENTATION/main2_eval_results.json
```

Analyze the saved results:

```bash
python3 implementations/MY_IMPLEMENTATION/analyze_main2_eval.py --results-path implementations/MY_IMPLEMENTATION/main2_eval_results.json --dataset-path evals/ground-truth/ground_truth_om_handbook_v3.json --sample-limit 10
```

## Environment Setup

The pipeline uses `load_dotenv()`, so `.env` is the main source of configuration.

### Vertex AI / Google Cloud

- `VERTEX_AI_DATASTORE_ID` is required.
- `GOOGLE_CLOUD_LOCATION` is read by `Configs` and shown in the CLI output.
- `VERTEX_QA_QUERY` is optional and can replace `--query` for `main2.py`.
- `VERTEX_SEARCH_PAGE_SIZE` tunes the retrieval page size.
- `VERTEX_SEARCH_MAX_PAGES` tunes how many pages are fetched.
- `VERTEX_MAX_EVIDENCE_SENTENCES` limits selected evidence sentences.
- `VERTEX_MAX_RANKED_CHUNKS` limits the ranked chunk pool.
- `VERTEX_CITATION_DEBUG` enables extra citation debug logging.

### Langfuse

- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` enable tracing.
- `LANGFUSE_BASE_URL` is preferred by the tracing helper if present.
- `LANGFUSE_HOST` is also supported and is used by `Configs`.
- `LANGFUSE_ENABLED` can be set to `false` to disable tracing even when keys exist.
- `APP_ENV` is attached to trace metadata and Langfuse environment.
- `APP_VERSION` is attached to trace metadata and Langfuse release.
- `LANGFUSE_CAPTURE_FULL_PROMPT` controls whether the full prompt text is included in trace metadata.
- `LANGFUSE_USER_ID` and `LANGFUSE_SESSION_ID` can be used to group traces.

Current `.env` already includes `VERTEX_AI_DATASTORE_ID`, `GOOGLE_CLOUD_LOCATION`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.

## Commands

### One query through `main2.py`

```bash
python3 implementations/MY_IMPLEMENTATION/main2.py --query "<your question>"
```

If you want to pass the query through the environment instead:

```bash
VERTEX_QA_QUERY="<your question>" python3 implementations/MY_IMPLEMENTATION/main2.py
```

### Full evaluation with `evaluate_main2.py`

The default `--max-items` is 5. To run the full 100-item ground-truth set, use:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --max-items 100 --output-path implementations/MY_IMPLEMENTATION/main2_eval_results.json
```

Useful narrower variants:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --item-id omh-fact-002
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --category procedural --max-items 20
```

Note: `--max-items 0` currently means "no limit" because the code only slices when `max_items > 0`.

### Regenerate `main2_eval_results.json`

Use the same evaluation command above, pointed at the checked-in output path:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --max-items 100 --output-path implementations/MY_IMPLEMENTATION/main2_eval_results.json
```

### Run `analyze_main2_eval.py`

```bash
python3 implementations/MY_IMPLEMENTATION/analyze_main2_eval.py --results-path implementations/MY_IMPLEMENTATION/main2_eval_results.json --dataset-path evals/ground-truth/ground_truth_om_handbook_v3.json --sample-limit 10
```

## Evaluation Workflow

1. `evaluate_main2.py` loads the dataset from [evals/ground-truth/ground_truth_om_handbook_v3.json](evals/ground-truth/ground_truth_om_handbook_v3.json).
2. Each item provides a `question`, `category`, `expected_behavior`, `expected_answer`, and `traceability` metadata.
3. For each item, the evaluator calls `search_discovery_engine(question, config, credentials)` from [main2.py](implementations/MY_IMPLEMENTATION/main2.py).
4. The raw pipeline response is normalized into the evaluator contract:
   - `output.text` becomes the generated answer text.
   - `output.safety_level` is the pipeline safety label.
   - `output.sources` is the normalized citation/source list.
5. The evaluators score three axes:
   - answer correctness and answer similarity
   - safety level match
   - traceability recall, precision, and completeness
6. The evaluation script stores one result row per item and a top-level summary.
7. The analyzer reads the saved JSON and prints grouped metrics and failure cohorts.

### How `main2.py` answers a query

`main2.py` is a deterministic retrieval-and-synthesis pipeline, not a free-form chat model. It:

- classifies the query
- retrieves chunks from Vertex AI Search / Discovery Engine
- normalizes and ranks evidence
- assembles an answer from the selected evidence
- formats citations and sources
- returns a structured response

The pipeline also writes Langfuse spans around those stages.

## What `main2_eval_results.json` Contains

The current file is the output of `evaluate_main2.py`, not the raw `main2.py` response.

### Top-level structure

- `summary`: aggregate evaluation metrics and counts.
- `results`: per-item evaluation rows.
- `langfuse_trace_id`: the evaluation-run trace ID.
- `langfuse_observation_id`: the evaluation-run observation ID.

### `summary` fields

- `item_count`
- `search_status_counts`
- `insufficient_evidence_count`
- `metric_means`

`metric_means` currently includes:
- `answer_similarity`
- `answer_correctness`
- `safety_level_match`
- `traceability_recall`
- `traceability_precision`
- `traceability_complete`

### Per-item fields in `results[]`

Each item row currently contains:

- `id`
- `question`
- `category`
- `expected_behavior`
- `expected_safety_level`
- `search_status`
- `search_error`
- `insufficient_evidence`
- `raw_citation_count`
- `source_count`
- `output`
- `evaluations`
- `evaluation_metrics`
- `missing_required_terms`
- `failure_buckets`
- `pipeline_trace_id`
- `pipeline_observation_id`
- `langfuse_trace_id`
- `langfuse_observation_id`
- `evaluation_trace_id`
- `evaluation_observation_id`

### `output` fields

The evaluator stores a normalized output object shaped like:

- `text`: generated answer text
- `safety_level`: pipeline safety label
- `sources`: normalized citation sources

Each source may include:
- `document_name`
- `document_id`
- `page`
- `section_heading`
- `chunk_id`
- `source_text`

### Expected answer and required facts

The expected answer is not duplicated in the results row. It lives in the dataset file under each item:

- `expected_answer.text`
- `expected_answer.acceptable_variants`
- `expected_answer.must_include`
- `expected_answer.must_not_include`
- `traceability`

## Langfuse Layout

### PDF QA traces

`main2.py` emits a pipeline span named `pdf_qa_pipeline` and child observations for:

- `classify_request`
- `vertex_retrieval`
- `normalize_evidence`
- `build_prompt`
- `generate_answer`
- `parse_model_output`
- `format_citations`
- `assemble_response`

When you run `main2.py` directly, `pdf_qa_pipeline` is typically the trace root.

When you run `evaluate_main2.py`, `pdf_qa_pipeline` runs inside `evaluate_item`, so it is part of the `evaluation_run` trace tree. In that mode, use `pipeline_observation_id` from JSON for direct navigation to the pipeline step.

Inspect the trace tree to see the retrieved chunks, the selected evidence, the answer text, and the citation summary.

### Evaluation run traces

`evaluate_main2.py` emits a root trace named `evaluation_run` and child observations for:

- `evaluate_item`

The evaluation run trace stores aggregate summary data and aggregate metric means, and emits aggregate scores named:

- `mean_answer_similarity`
- `mean_answer_correctness`
- `mean_safety_level_match`
- `mean_traceability_recall`
- `mean_traceability_precision`
- `mean_traceability_complete`

### Analysis traces

`analyze_main2_eval.py` emits a root trace named `analysis_run` plus helper observations such as:

- `load_results`
- `compute_cohorts`

### What to inspect in each trace

- root trace metadata for `APP_ENV`, `APP_VERSION`, and datastore context
- `classify_request` for query safety / scope decisions
- `vertex_retrieval` for retrieved chunk previews and raw retrieval counts
- `normalize_evidence` for selected evidence sentences, ranks, and anchors
- `build_prompt` for the prompt summary and evidence list
- `generate_answer` for the synthesized output payload
- `format_citations` for citation IDs, section headings, and page numbers
- `evaluate_item` for per-item evaluator outputs
- per-item scores on `evaluate_item` observations
- trace-level summary scores for aggregate evaluation metrics

### Metadata currently attached

Pipeline metadata (compact, truncated):

- query preview and query length
- datastore ID/project/short ID and location
- app env/version
- decision type and final status
- safety level
- source/citation counts
- unknown section heading count
- error stage/type/message when failures happen

Evaluation run metadata:

- dataset path
- output path
- max items
- item count
- run timestamp
- app env/version
- summary counts/means after completion

Evaluate item metadata:

- item ID/category/expected behavior/expected safety
- search status/error
- source/citation counts
- insufficient evidence
- metric values
- missing required terms
- failure buckets
- pipeline trace/observation IDs

### Generation, latency, and cost

`generate_answer` currently wraps a deterministic synthesis step rather than a real LLM call. You should expect useful timing and payload inspection, but token and cost data may be absent or not meaningful.

## Recommended Filters and Searches in Langfuse

Use these first:

- trace name `pdf_qa_pipeline`
- trace name `evaluation_run`
- trace name `analysis_run`
- tag `pdf_qa`
- tag `vertex_search`
- tag `evaluation`
- tag `summary`
- tags like `category:factual`, `category:procedural`, `category:table`
- tags like `expected_behavior:answer`, `expected_behavior:refuse_unsafe`, `expected_behavior:refuse_out_of_scope`
- tags like `search_status:success`, `search_status:error`
- tags like `source_count:0`, `low_correctness`, `low_traceability`, `table_question`, `numeric_question`
- failure-bucket tags like `low_correctness_high_similarity`, `no_sources`, `search_error`, `safety_mismatch`
- `APP_ENV` metadata if you run multiple environments
- trace ID copied from `main2_eval_results.json`

Score filters to use:

- low `answer_similarity`
- low `traceability_recall`
- low `traceability_precision`
- low `mean_answer_correctness`
- low `mean_safety_level_match`

Current limitations:

- `evaluation_run` uses one trace for many items; category/search tags are updated repeatedly during the run, so per-item filtering is most reliable via `evaluate_item` metadata and scores.
- pipeline and evaluation may share the same trace ID during evaluation runs; use `pipeline_observation_id` to jump to the exact pipeline observation.

## Metric Definitions

These follow the evaluator code in [implementations/handbook_qa/evaluators.py](implementations/handbook_qa/evaluators.py).

- `answer_similarity`: best semantic similarity between the generated answer text and the expected answer text or an acceptable variant.
- `answer_correctness`: `true` only when similarity clears the threshold and all `must_include` terms are present and all `must_not_include` terms are absent.
- `safety_level_match`: exact match between the emitted `safety_level` and the expected level.
- `traceability_recall`: fraction of required grounding sources that were cited.
- `traceability_precision`: fraction of cited sources that match any expected source.
- `traceability_complete`: `true` only when every required source was cited.

## How to Debug a Failed Item

1. Open `implementations/MY_IMPLEMENTATION/main2_eval_results.json`.
2. Find the failed row by `id`, `question`, or `category`.
3. Copy `pipeline_trace_id` and `pipeline_observation_id` (and optionally `evaluation_trace_id`).
4. Open the trace in Langfuse and navigate to the `pipeline_observation_id` / `pdf_qa_pipeline` observation.
5. Check `classify_request` to see whether the query was treated as unsafe or out of scope.
6. Check `vertex_retrieval` to see what chunks came back and whether the raw retrieval was broad or narrow.
7. Check `normalize_evidence` to see which evidence sentences survived ranking and why.
8. Check `build_prompt` to see what evidence was actually fed into the synthesis step.
9. Check `generate_answer` and `parse_model_output` to compare the answer text against the expected required facts.
10. Check `format_citations` to see whether the cited sources and section headings line up with the required grounding set.
11. Check the evaluation metrics in the JSON row and the per-item observation scores to see which score failed and why.

### Common Failure Patterns

- High `answer_similarity` but low `answer_correctness`: the answer is semantically close but misses an exact required term or includes a forbidden term.
- `traceability_recall` is `0`: the cited sources do not match the required grounding sources.
- Low `traceability_precision`: the answer cites extra irrelevant sources.
- Low `safety_level_match`: the emitted `safety_level` does not match the expected label.
- `source_count` is `0`: the answer has no normalized sources, usually because retrieval produced no usable evidence.
- `search_status` is `error`: the retrieval step failed before evidence could be assembled.
- Table or numeric questions fail: inspect evidence ranking and citation selection; these are the most sensitive to missing exact facts or wrong section selection.
- Citation section heading is `Unknown`: the section parser could not infer a stable heading from the source metadata.
- The answer cites irrelevant evidence: evidence ranking is too broad or the selected sources are not specific enough.
- The model answers despite insufficient evidence: the evidence gate did not trigger a refusal path; inspect selected evidence and the answer assembly logic.

## Troubleshooting Checklist

- Confirm `.env` contains `VERTEX_AI_DATASTORE_ID`.
- Confirm Langfuse keys are present and valid.
- Confirm `LANGFUSE_ENABLED` is not set to `false`.
- Confirm the dataset path and results path are correct.
- Confirm the trace ID in the JSON row matches the trace you opened in Langfuse.
- Check the evaluator comment strings in `evaluations[]` for the exact reason a score failed.
- If you changed the pipeline and the JSON does not contain trace IDs, rerun `evaluate_main2.py` to regenerate it.
- If pipeline and evaluation share one trace ID, use `pipeline_observation_id` to jump to the right node.
- Use `failure_buckets` and `missing_required_terms` in JSON to quickly shortlist candidates before opening Langfuse.

## Validation Commands

Run one query with Langfuse enabled:

```bash
python3 implementations/MY_IMPLEMENTATION/main2.py --query "What is the minimum required capacity and rating of the portable fire extinguisher each bulk plant must have?"
```

Run one query with Langfuse disabled:

```bash
LANGFUSE_ENABLED=false python3 implementations/MY_IMPLEMENTATION/main2.py --query "What is the minimum required capacity and rating of the portable fire extinguisher each bulk plant must have?"
```

Run a single evaluation item:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --item-id omh-fact-002 --output-path /tmp/main2_eval_single_item.json
```

Run a category slice:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --category table --max-items 3 --output-path /tmp/main2_eval_table_slice.json
```

Run a full evaluation:

```bash
python3 implementations/MY_IMPLEMENTATION/evaluate_main2.py --max-items 100 --output-path implementations/MY_IMPLEMENTATION/main2_eval_results.json
```

Run analysis:

```bash
python3 implementations/MY_IMPLEMENTATION/analyze_main2_eval.py --results-path implementations/MY_IMPLEMENTATION/main2_eval_results.json --dataset-path evals/ground-truth/ground_truth_om_handbook_v3.json --sample-limit 8
```

## TODOs and Gaps

- TODO: the deterministic `generate_answer` step is not a real LLM call, so token and cost reporting is limited.
- TODO: if you want perfect per-item trace-level tag filtering, create dedicated traces per item instead of a single multi-item `evaluation_run` trace.
