# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project Context

- **Program:** Vector Institute **Agentic AI Evaluation Bootcamp** (`agentic-ai-evaluation-bootcamp-202602`).
- **Team:** 4 members working on a single capstone use case.
- **Use case:** **Grounded QA over an industrial handbook.** The agent answers
  natural-language questions and must ground every answer in the source
  document, citing the specific page/section it relied on.
- **Source document:** *Operations and Maintenance Handbook for LP-Gas Bulk
  Storage Facilities* (built around NFPA 58 / NFPA 10 / NFPA 25 and related DOT
  regulations). It is a safety-critical procedures manual, so **accuracy and
  traceability matter more than fluency** — a confidently wrong or
  ungrounded answer is a failure, not a minor defect.
- **Baseline:** Our implementation is derived from the
  [`knowledge_qa`](implementations/knowledge_qa/README.md) reference
  implementation. The key difference: instead of grounding on **live web
  search**, we ground on a **Vertex AI Search data store** built from our own
  handbook PDF.

## Repository Layout (what's ours vs. provided)

- [data/raw/](data/raw) — the source PDF (`om handbook.pdf`). Treat as
  read-only input.
- [data/processed/om_handbook/](data/processed/om_handbook) — the ingested
  handbook: `chunks.jsonl` (vector-store input), plus `sections.jsonl`,
  `paragraphs.json`, `tables/`, `content.md`, etc. `chunks.jsonl` is the
  artifact consumed by the data store builder.
- [evals/ground-truth/](evals/ground-truth) — **our** evaluation dataset.
  - [schema.json](evals/ground-truth/schema.json) — JSON Schema for the
    ground-truth dataset (authoritative contract).
  - [example.json](evals/ground-truth/example.json) — a small, illustrative
    example that conforms to the schema. Use it as the template when authoring
    real ground-truth items.
- [implementations/handbook_qa/](implementations/handbook_qa) — **the home for
  our agent implementation** (currently empty). Mirror the structure of
  [`implementations/knowledge_qa/`](implementations/knowledge_qa) when building
  it out.
- [implementations/knowledge_qa/](implementations/knowledge_qa) — the baseline
  reference. **Read this first**; copy its patterns rather than inventing new
  ones.
- [scripts/create_handbook_datastore.py](scripts/create_handbook_datastore.py)
  — provisions the GCS bucket, transforms/uploads `chunks.jsonl`, creates a
  `CONTENT_REQUIRED` Vertex AI Search data store, imports docs, and waits for
  indexing. Prints the `VERTEX_AI_DATASTORE_ID` to add to `.env`.
- [aieng-eval-agents/](aieng-eval-agents) — the shared `aieng` package
  (agent base classes, evaluation harness, graders, tools, configs). Prefer
  reusing these utilities over writing new ones.

## Tech Stack & Conventions

- **Python 3.12+**, managed with **`uv`**. Run commands with
  `uv run --env-file .env ...`.
- **Agent framework:** Google ADK (the baseline uses a PlanReAct architecture
  with a ReAct loop per step).
- **Grounding/retrieval:** Vertex AI Search (Discovery Engine) data store built
  from `chunks.jsonl`.
- **Tracing & experiments:** Langfuse.
- **Config:** Pydantic settings; all secrets/IDs via `.env`
  (use `os.getenv(...)`, never hard-code). Keep `.env.example` in sync.
- Follow [GUIDELINES.md](GUIDELINES.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Building / Refreshing the Data Store

From the repo root, after `gcloud auth application-default login`:

```bash
uv run python -m scripts.create_handbook_datastore \
    --bucket agentic-ai-evaluation-bootcamp-hitachi-rail \
    --datastore-id om-handbook-v1 \
    --input data/processed/om_handbook/chunks.jsonl
```

Then copy the printed `VERTEX_AI_DATASTORE_ID` into `.env`.

## Evaluation Model (how we grade)

Defined by [evals/ground-truth/schema.json](evals/ground-truth/schema.json).
Each ground-truth item is graded on three axes:

1. **Answer correctness** — `expected_answer.text` compared via **fuzzy /
   semantic similarity** (default `embedding_cosine`, threshold ~0.8). Supports
   `acceptable_variants`, `must_include` (hard gate), and `must_not_include`
   (hallucination/unsafe-content gate).
2. **Safety level** — `safety_level`
   (`public`→`internal`→`sensitive`→`restricted`→`prohibited`), graded as an
   **exact match**.
3. **Traceability** — `traceability[]` source(s) (`document_name`,
   `document_id`, `page`, `section_heading`, optional `chunk_id`), graded as
   **exact matches**. A correct answer must cite at least the `required`
   sources. This is the core of "grounded" QA.

When adding ground-truth items, validate against the schema and keep
`page` / `section_heading` / `chunk_id` aligned with the actual entries in
`data/processed/om_handbook/`.

## Working Agreements for Agents

- **Read before writing:** consult the `knowledge_qa` baseline and the shared
  `aieng` package before adding code; reuse existing patterns and utilities.
- **Grounding is the product:** never let the agent answer from parametric
  knowledge when the handbook is silent — prefer "not found in the handbook"
  over a plausible guess. Always surface the supporting source.
- **Don't break the eval contract:** changes to ground-truth files must stay
  schema-valid. If the schema must change, update
  [schema.json](evals/ground-truth/schema.json),
  [example.json](evals/ground-truth/example.json), and any grader code together.
- **Treat `data/raw/` and processed artifacts as inputs**, not scratch space.
  Regenerate the data store rather than hand-editing `chunks.jsonl`.
- **Secrets live in `.env`.** Do not commit keys, datastore IDs, or bucket
  names beyond what already exists in tracked config.
- Keep new implementation code under
  [implementations/handbook_qa/](implementations/handbook_qa).
