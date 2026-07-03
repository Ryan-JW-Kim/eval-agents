import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aieng.agent_evals.configs import Configs
from implementations.MY_IMPLEMENTATION.main2 import get_credentials, search_discovery_engine
from implementations.MY_IMPLEMENTATION.langfuse_tracing import (
    compact_text,
    safe_create_score,
    safe_score_observation,
    get_current_observation_id,
    get_current_trace_id,
    get_langfuse_settings,
    langfuse_span,
    safe_flush_langfuse,
    safe_score_trace,
    safe_update_observation,
    safe_update_trace,
    truncate_value,
    utc_now_iso,
)
from implementations.handbook_qa.evaluators import (
    answer_correctness_evaluator,
    safety_level_evaluator,
    traceability_evaluator,
)


def _extract_document_id_from_uri(uri: str) -> str:
    parts = uri.split("/")
    if "documents" in parts:
        idx = parts.index("documents")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _canonical_document_id(document_id: str, doc_name: str) -> str:
    lowered = f"{document_id} {doc_name}".lower()
    if "om_handbook" in lowered or "om handbook" in lowered:
        return "doc_om_handbook_perc"
    return document_id


def _normalize_document_name(doc_name: str, document_id: str) -> str:
    lowered = f"{doc_name} {document_id}".lower()
    if "om_handbook" in lowered or "om handbook" in lowered:
        return "om handbook.pdf"
    return doc_name or document_id


def _to_int_or_none(value: Any) -> int | None:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _tokenize(text: str) -> set[str]:
    return set(part.lower() for part in re.findall(r"[a-zA-Z0-9]+", text))


def _section_from_page(page: int | None) -> str:
    if page is None:
        return "Unknown"
    if page <= 14:
        return "Section 1 - Introduction"
    if page <= 18:
        return "Section 2 - Employee Safety"
    if page <= 30:
        return "Section 3 - Emergency Procedures Plan"
    if page <= 74:
        return "Section 4 - General Operations and Safety Requirements"
    if page <= 108:
        return "Section 5 - Plant Operations Procedures"
    if page <= 141:
        return "Section 6 - General Maintenance and Inspection Requirements"
    if page <= 167:
        return "Section 7 - Maintenance & Inspection Checklist Procedure"
    if page <= 176:
        return "Section 8 - Maintenance of Fire Protection Equipment"
    return "Unknown"


def _normalize_section_heading(section_heading: str, page: int | None) -> str:
    heading = section_heading.strip()
    if heading and heading.lower() != "unknown":
        section_match = re.search(r"(Section\s+\d+\s*-\s*[^>\n]+)", heading, flags=re.IGNORECASE)
        if section_match:
            return section_match.group(1).strip()
        appendix_match = re.search(r"(Appendix\s+[A-Za-z0-9]+\s*-\s*[^>\n]+)", heading, flags=re.IGNORECASE)
        if appendix_match:
            return appendix_match.group(1).strip()
        return heading
    return _section_from_page(page)


def _augment_text_for_eval(text: str) -> str:
    aliases: list[str] = []

    for number in re.findall(r"\b(\d+)\s+pounds?\b", text, flags=re.IGNORECASE):
        aliases.append(f"{number} lb")
    for number in re.findall(r"\b(\d+)\s+lb\b", text, flags=re.IGNORECASE):
        aliases.append(f"{number} pounds")
    for number in re.findall(r"\b(\d+)\s+ft\b", text, flags=re.IGNORECASE):
        aliases.append(f"{number} feet")

    if not aliases:
        return text

    alias_text = " ".join(dict.fromkeys(aliases))
    return f"{text} {alias_text}".strip()


def _citation_support_score(citation: dict[str, Any]) -> int:
    raw = citation.get("support_score", 0)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return 0


def _build_output_for_evaluators(search_result: dict[str, Any], question: str) -> dict[str, Any]:
    summary_text = str(search_result.get("summary", "")).strip()
    safety_level = str(search_result.get("safety_level") or "negligible")

    if search_result.get("status") == "error":
        summary_text = "Unable to retrieve enough handbook evidence to answer this question."

    constraints_raw = search_result.get("constraints", [])
    if isinstance(constraints_raw, list):
        constraints = [str(c).strip() for c in constraints_raw if str(c).strip()]
        if constraints:
            summary_text = f"{summary_text} Constraints: {'; '.join(constraints[:3])}.".strip()

    citations = search_result.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    relevance_tokens = _tokenize(f"{question} {summary_text}")
    ranked_sources: list[tuple[int, dict[str, Any]]] = []
    seen_source_keys: set[tuple[str, int | None, str]] = set()

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        uri = str(citation.get("uri", ""))
        page = _to_int_or_none(citation.get("page_number"))

        document_id = _extract_document_id_from_uri(uri)
        doc_name = _normalize_document_name(str(citation.get("doc_name", "")).strip(), document_id)
        document_id = _canonical_document_id(document_id, doc_name)

        section_heading = _normalize_section_heading(str(citation.get("section_title", "")).strip(), page)
        chunk_id = str(citation.get("paragraph_number", "")).strip()
        source_text = str(citation.get("source_text", citation.get("snippet", "")))

        source = {
            "document_name": doc_name,
            "document_id": document_id,
            "page": page,
            "section_heading": section_heading,
            "chunk_id": chunk_id,
            "source_text": source_text,
        }
        source = {k: v for k, v in source.items() if v not in ("", None)}

        key = (
            str(source.get("document_name", "")).lower(),
            source.get("page") if isinstance(source.get("page"), int) else None,
            str(source.get("section_heading", "")).lower(),
        )
        if not source or key in seen_source_keys:
            continue

        seen_source_keys.add(key)
        snippet = str(citation.get("snippet", ""))
        overlap = len(_tokenize(snippet) & relevance_tokens)
        support = _citation_support_score(citation)
        ranked_sources.append((overlap + support, source))

    ranked_sources.sort(key=lambda item: item[0], reverse=True)
    max_sources = 3
    sources = [source for _, source in ranked_sources[:max_sources]]

    return {
        "text": _augment_text_for_eval(summary_text),
        "safety_level": safety_level,
        "sources": sources,
    }


def _item_metadata(item: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    expected_answer = item.get("expected_answer", {})
    if not isinstance(expected_answer, dict):
        expected_answer = {}

    answer_match = dict(defaults.get("answer_match", {}))
    item_answer_match = item.get("answer_match", {})
    if isinstance(item_answer_match, dict):
        answer_match.update(item_answer_match)

    return {
        "acceptable_variants": expected_answer.get("acceptable_variants", []) or [],
        "must_include": expected_answer.get("must_include", []) or [],
        "must_not_include": expected_answer.get("must_not_include", []) or [],
        "answer_match": answer_match,
        "safety_level": item.get("safety_level"),
        "traceability": item.get("traceability", []) or [],
    }


def _evaluate_one_item(
    *,
    item: dict[str, Any],
    config: Configs,
    credentials: Any,
    defaults: dict[str, Any],
    evaluation_run_trace_id: str | None,
    evaluation_run_observation_id: str | None,
) -> dict[str, Any]:
    question = str(item.get("question", "")).strip()
    item_id = str(item.get("id", "")).strip()
    category = str(item.get("category", "")).strip()
    expected_behavior = str(item.get("expected_behavior", "")).strip() or "answer"
    expected_safety_level = str(item.get("safety_level", "")).strip() or "unknown"

    expected_answer = item.get("expected_answer", {})
    expected_text = ""
    if isinstance(expected_answer, dict):
        expected_text = str(expected_answer.get("text", ""))

    try:
        search_result = search_discovery_engine(
            question,
            config,
            credentials,
            trace_context={
                "evaluation_item_id": item_id,
                "evaluation_category": category,
                "evaluation_expected_behavior": expected_behavior,
                "evaluation_run_trace_id": evaluation_run_trace_id,
                "evaluation_run_observation_id": evaluation_run_observation_id,
            },
        )
        output = _build_output_for_evaluators(search_result, question)
        metadata = _item_metadata(item, defaults)

        evaluations = []
        evaluations.extend(
            answer_correctness_evaluator(
                output=output,
                expected_output=expected_text,
                metadata=metadata,
            )
        )
        evaluations.append(safety_level_evaluator(output=output, metadata=metadata))
        evaluations.extend(traceability_evaluator(output=output, metadata=metadata))
    except Exception as exc:  # noqa: BLE001
        output = {
            "text": "Unable to evaluate this item due to an internal evaluation error.",
            "safety_level": "unknown",
            "sources": [],
        }
        search_result = {
            "status": "error",
            "error": f"evaluation stage failed: {type(exc).__name__}: {exc}",
            "insufficient_evidence": True,
            "citations": [],
            "langfuse_trace_id": None,
            "langfuse_observation_id": None,
        }
        evaluations = []

    serializable_evals = [
        {
            "name": e.name,
            "value": e.value,
            "data_type": e.data_type,
            "comment": e.comment,
        }
        for e in evaluations
    ]

    citations = search_result.get("citations", [])
    source_count = len(output.get("sources", [])) if isinstance(output.get("sources"), list) else 0
    evaluation_metrics: dict[str, Any] = {}
    for evaluation in evaluations:
        if isinstance(evaluation.value, (bool, int, float)):
            evaluation_metrics[str(evaluation.name)] = evaluation.value

    pipeline_trace_id = search_result.get("langfuse_trace_id")
    pipeline_observation_id = search_result.get("langfuse_observation_id")

    result = {
        "id": item_id,
        "question": question,
        "category": category,
        "expected_behavior": expected_behavior,
        "expected_safety_level": expected_safety_level,
        "search_status": search_result.get("status", "unknown"),
        "search_error": search_result.get("error"),
        "insufficient_evidence": bool(search_result.get("insufficient_evidence", False)),
        "raw_citation_count": len(citations) if isinstance(citations, list) else 0,
        "source_count": source_count,
        "output": output,
        "evaluations": serializable_evals,
        "evaluation_metrics": evaluation_metrics,
        "missing_required_terms": _missing_required_terms(serializable_evals),
        "pipeline_trace_id": pipeline_trace_id,
        "pipeline_observation_id": pipeline_observation_id,
        "langfuse_trace_id": get_current_trace_id(),
        "langfuse_observation_id": get_current_observation_id(),
        "evaluation_trace_id": get_current_trace_id(),
        "evaluation_observation_id": get_current_observation_id(),
    }
    result["failure_buckets"] = _failure_buckets(result)

    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    search_status_counts: dict[str, int] = defaultdict(int)
    insufficient_evidence_count = 0

    for result in results:
        search_status = str(result.get("search_status", "unknown"))
        search_status_counts[search_status] += 1

        if result.get("insufficient_evidence"):
            insufficient_evidence_count += 1

        for evaluation in result.get("evaluations", []):
            name = evaluation.get("name")
            value = evaluation.get("value")
            if not isinstance(name, str) or not name:
                continue
            if isinstance(value, bool):
                values_by_metric[name].append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                values_by_metric[name].append(float(value))

    metric_means: dict[str, float] = {}
    for name, values in values_by_metric.items():
        if values:
            metric_means[name] = sum(values) / len(values)

    return {
        "item_count": len(results),
        "search_status_counts": dict(search_status_counts),
        "insufficient_evidence_count": insufficient_evidence_count,
        "metric_means": metric_means,
    }


def _missing_required_terms(evaluations: list[dict[str, Any]]) -> list[str]:
    for evaluation in evaluations:
        if evaluation.get("name") != "answer_correctness":
            continue
        comment = str(evaluation.get("comment", ""))
        marker = "missing required:"
        if marker not in comment:
            return []
        tail = comment.split(marker, 1)[1].strip()
        terms = re.findall(r"'([^']+)'", tail)
        return terms if terms else ([tail] if tail else [])
    return []


def _failure_buckets(result: dict[str, Any]) -> list[str]:
    buckets: list[str] = []
    metrics = result.get("evaluation_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    similarity = float(metrics.get("answer_similarity", 0.0) or 0.0)
    correctness = bool(metrics.get("answer_correctness", False))
    recall = float(metrics.get("traceability_recall", 0.0) or 0.0)
    precision = float(metrics.get("traceability_precision", 0.0) or 0.0)
    safety_match = bool(metrics.get("safety_level_match", False))

    search_status = str(result.get("search_status", "unknown"))
    source_count = int(result.get("source_count", 0) or 0)
    category = str(result.get("category", "")).strip().lower()
    expected_behavior = str(result.get("expected_behavior", "answer")).strip().lower()

    if similarity >= 0.8 and not correctness:
        buckets.append("low_correctness_high_similarity")
    if source_count == 0:
        buckets.append("no_sources")
    if search_status == "error":
        buckets.append("search_error")
    if recall == 0.0 or precision < 0.2:
        buckets.append("low_traceability")
    if not safety_match:
        buckets.append("safety_mismatch")
    if category == "table" or category == "numeric":
        buckets.append("table_numeric_failure")
    if expected_behavior == "refuse" and correctness is False:
        buckets.append("refusal_failure")

    if not buckets:
        buckets.append("none")
    return buckets


def _item_tags(result: dict[str, Any], expected_safety_level: str) -> list[str]:
    metrics = result.get("evaluation_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    category = str(result.get("category", "") or "unknown").strip().lower() or "unknown"
    expected_behavior = str(result.get("expected_behavior", "answer") or "answer").strip().lower() or "answer"
    search_status = str(result.get("search_status", "unknown") or "unknown").strip().lower() or "unknown"
    source_count = int(result.get("source_count", 0) or 0)

    tags = [
        "evaluation",
        "pdf_qa",
        f"category:{category}",
        f"expected_behavior:{expected_behavior}",
        f"search_status:{search_status}",
        f"expected_safety:{re.sub(r'[^a-z0-9_\-]+', '_', expected_safety_level.strip().lower() or 'unknown')}",
    ]

    if source_count == 0:
        tags.append("source_count:0")
    if bool(result.get("insufficient_evidence", False)):
        tags.append("insufficient_evidence")

    if not bool(metrics.get("answer_correctness", False)):
        tags.append("low_correctness")
    if float(metrics.get("traceability_recall", 0.0) or 0.0) == 0.0 or float(metrics.get("traceability_precision", 0.0) or 0.0) < 0.2:
        tags.append("low_traceability")
    if category == "table":
        tags.append("table_question")
    if re.search(r"\b(how many|how far|how close|minimum|maximum|percent|years?)\b", str(result.get("question", "")), flags=re.IGNORECASE):
        tags.append("numeric_question")

    for bucket in _failure_buckets(result):
        if bucket != "none":
            tags.append(bucket)

    # Preserve order and drop duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def _evaluation_trace_metadata(dataset_path: str, output_path: str, max_items: int, item_count: int) -> dict[str, Any]:
    settings = get_langfuse_settings()
    return {
        "app_env": settings.app_env,
        "app_version": settings.app_version,
        "dataset_path": dataset_path,
        "output_path": output_path,
        "max_items": max_items,
        "item_count": item_count,
        "run_timestamp": utc_now_iso(),
    }


def _load_dataset(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise click.ClickException(f"Dataset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Dataset file is not valid JSON: {path}: {exc}") from exc


@click.command()
@click.option(
    "--dataset-path",
    default="evals/ground-truth/ground_truth_om_handbook_v3.json",
    show_default=True,
    help="Path to ground-truth dataset JSON.",
)
@click.option("--item-id", default=None, help="Evaluate only this item id, e.g. omh-fact-003.")
@click.option(
    "--category",
    default=None,
    help="Evaluate only a category (factual, procedural, table, synthesis, out_of_scope, adversarial).",
)
@click.option("--max-items", default=5, show_default=True, type=int, help="Maximum number of items to evaluate.")
@click.option(
    "--output-path",
    default="implementations/MY_IMPLEMENTATION/main2_eval_results.json",
    show_default=True,
    help="Where to write detailed evaluation results JSON.",
)
def cli(dataset_path: str, item_id: str | None, category: str | None, max_items: int, output_path: str) -> None:
    load_dotenv()

    credentials = get_credentials()
    config = Configs()  # type: ignore[call-arg]

    if not config.vertex_datastore_id:
        raise click.ClickException("VERTEX_AI_DATASTORE_ID is not set.")

    dataset = _load_dataset(Path(dataset_path))
    items = dataset.get("items", [])
    if not isinstance(items, list):
        raise click.ClickException("Dataset has invalid 'items' format.")

    typed_items = [item for item in items if isinstance(item, dict)]

    if item_id:
        typed_items = [item for item in typed_items if item.get("id") == item_id]
        if not typed_items:
            raise click.ClickException(f"Item id not found: {item_id}")

    if category:
        category_norm = category.strip().lower()
        typed_items = [
            item
            for item in typed_items
            if str(item.get("category", "")).strip().lower() == category_norm
        ]
        if not typed_items:
            raise click.ClickException(f"Category not found or empty: {category}")

    if max_items > 0:
        typed_items = typed_items[:max_items]

    defaults = dataset.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    trace_metadata = _evaluation_trace_metadata(dataset_path, output_path, max_items, len(typed_items))
    run_trace_id: str | None = None
    run_observation_id: str | None = None

    with langfuse_span(
        name="evaluation_run",
        input={"dataset_path": dataset_path, "item_count": len(typed_items), "max_items": max_items},
        metadata=trace_metadata,
    ) as run_span:
        safe_update_trace(name="evaluation_run", metadata=trace_metadata, tags=["evaluation", "pdf_qa", "evaluation_run"])
        run_trace_id = get_current_trace_id()
        run_observation_id = get_current_observation_id()

        results: list[dict[str, Any]] = []
        for index, item in enumerate(typed_items, start=1):
            click.echo(f"Evaluating {item.get('id', '<unknown>')} ...")
            question = str(item.get("question", "")).strip()
            expected_safety_level = str(item.get("safety_level", "")).strip() or "unknown"
            with langfuse_span(
                name="evaluate_item",
                input={
                    "item_id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "question_preview": compact_text(question, 240),
                    "position": index,
                },
                metadata={"item_id": item.get("id", ""), "category": item.get("category", ""), "position": index},
            ) as item_span:
                result = _evaluate_one_item(
                    item=item,
                    config=config,
                    credentials=credentials,
                    defaults=defaults,
                    evaluation_run_trace_id=run_trace_id,
                    evaluation_run_observation_id=run_observation_id,
                )
                results.append(result)

                item_tags = _item_tags(result, expected_safety_level)
                item_metadata = {
                    "item_id": result.get("id"),
                    "category": result.get("category"),
                    "expected_behavior": result.get("expected_behavior"),
                    "expected_safety_level": result.get("expected_safety_level"),
                    "search_status": result.get("search_status"),
                    "source_count": result.get("source_count"),
                    "raw_citation_count": result.get("raw_citation_count"),
                    "insufficient_evidence": result.get("insufficient_evidence"),
                    "failure_buckets": result.get("failure_buckets", []),
                    "missing_required_terms": result.get("missing_required_terms", []),
                    "pipeline_trace_id": result.get("pipeline_trace_id"),
                    "pipeline_observation_id": result.get("pipeline_observation_id"),
                    "item_tags": item_tags,
                }
                safe_update_observation(
                    item_span,
                    output=truncate_value(
                        {
                            "search_status": result.get("search_status"),
                            "search_error": result.get("search_error"),
                            "insufficient_evidence": result.get("insufficient_evidence"),
                            "source_count": result.get("source_count"),
                            "raw_citation_count": result.get("raw_citation_count"),
                            "evaluation_metrics": result.get("evaluation_metrics", {}),
                            "failure_buckets": result.get("failure_buckets", []),
                            "pipeline_trace_id": result.get("pipeline_trace_id"),
                        }
                    ),
                    metadata=truncate_value({**item_metadata, **result.get("evaluation_metrics", {})}),
                )

                metric_map = result.get("evaluation_metrics", {})
                if isinstance(metric_map, dict):
                    for metric_name, metric_value in metric_map.items():
                        if isinstance(metric_value, bool):
                            safe_score_observation(
                                item_span,
                                name=metric_name,
                                value=1.0 if metric_value else 0.0,
                                data_type="BOOLEAN",
                                comment=f"item {result.get('id', '')}",
                                metadata={"item_id": result.get("id", ""), "category": result.get("category", "")},
                            )
                        elif isinstance(metric_value, (int, float)):
                            safe_score_observation(
                                item_span,
                                name=metric_name,
                                value=float(metric_value),
                                data_type="NUMERIC",
                                comment=f"item {result.get('id', '')}",
                                metadata={"item_id": result.get("id", ""), "category": result.get("category", "")},
                            )

                        pipeline_trace_id = result.get("pipeline_trace_id")
                        pipeline_observation_id = result.get("pipeline_observation_id")
                        if isinstance(pipeline_trace_id, str) and pipeline_trace_id and isinstance(metric_value, (int, float, bool)):
                            score_kwargs: dict[str, Any] = {
                                "name": metric_name,
                                "value": float(metric_value) if isinstance(metric_value, (int, float, bool)) else metric_value,
                                "trace_id": pipeline_trace_id,
                                "comment": f"evaluation item {result.get('id', '')}",
                                "metadata": {
                                    "item_id": result.get("id", ""),
                                    "category": result.get("category", ""),
                                    "expected_behavior": result.get("expected_behavior", ""),
                                },
                            }
                            if isinstance(pipeline_observation_id, str) and pipeline_observation_id:
                                score_kwargs["observation_id"] = pipeline_observation_id
                            if isinstance(metric_value, bool):
                                score_kwargs["data_type"] = "BOOLEAN"
                                score_kwargs["value"] = 1.0 if metric_value else 0.0
                            else:
                                score_kwargs["data_type"] = "NUMERIC"
                            safe_create_score(**score_kwargs)

                safe_update_trace(
                    metadata={
                        "last_item_id": result.get("id"),
                        "last_item_category": result.get("category"),
                        "last_item_expected_behavior": result.get("expected_behavior"),
                        "last_item_search_status": result.get("search_status"),
                        "last_item_source_count": result.get("source_count"),
                        "last_item_failure_buckets": result.get("failure_buckets", []),
                    },
                    tags=item_tags,
                )

        summary = _summarize(results)
        metric_means = summary.get("metric_means", {})
        if isinstance(metric_means, dict):
            for metric_name, metric_value in metric_means.items():
                if isinstance(metric_value, (int, float)):
                    safe_score_trace(name=f"mean_{metric_name}", value=float(metric_value), comment="aggregate evaluation mean")

        safe_update_trace(output=truncate_value(summary), metadata={**trace_metadata, **summary}, tags=["evaluation", "summary", "evaluation_run"])
        safe_update_observation(run_span, output=truncate_value(summary), metadata={"summary_item_count": summary.get("item_count", 0)})

        output = {"summary": summary, "results": results, "langfuse_trace_id": run_trace_id, "langfuse_observation_id": run_observation_id}

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    click.echo("\nEvaluation summary:")
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"\nDetailed results written to: {out_path}")

    safe_flush_langfuse()


if __name__ == "__main__":
    cli()
