import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementations.MY_IMPLEMENTATION.langfuse_tracing import (
    get_current_observation_id,
    get_current_trace_id,
    get_langfuse_settings,
    langfuse_span,
    safe_flush_langfuse,
    safe_score_trace,
    safe_update_observation,
    safe_update_trace,
    truncate_value,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def _eval_map(result: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for evaluation in result.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        name = evaluation.get("name")
        if isinstance(name, str) and name:
            mapped[name] = evaluation.get("value")
    return mapped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_numeric_question(question: str) -> bool:
    return bool(re.search(r"\b(how many|how far|how close|minimum|maximum|percent|years?)\b", question, flags=re.IGNORECASE))


def _is_procedural_question(question: str) -> bool:
    return bool(re.search(r"\b(how|procedure|steps|during|before|after|if )\b", question, flags=re.IGNORECASE))


def _missing_required_terms(comment: str) -> list[str]:
    marker = "missing required:"
    if marker not in comment:
        return []
    tail = comment.split(marker, 1)[1].strip()
    # Common format: ['a', 'b', 'c']
    terms = re.findall(r"'([^']+)'", tail)
    if terms:
        return terms
    if tail:
        return [tail]
    return []


def _print_group_metrics(title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    print(f"\n== {title} ==")
    for group, rows in sorted(grouped.items()):
        print(
            group,
            len(rows),
            "ans_corr",
            round(_mean([float(bool(r.get("answer_correctness", False))) for r in rows]), 4),
            "ans_sim",
            round(_mean([float(r.get("answer_similarity", 0.0) or 0.0) for r in rows]), 4),
            "safety",
            round(_mean([float(bool(r.get("safety_level_match", False))) for r in rows]), 4),
            "trace_recall",
            round(_mean([float(r.get("traceability_recall", 0.0) or 0.0) for r in rows]), 4),
            "trace_precision",
            round(_mean([float(r.get("traceability_precision", 0.0) or 0.0) for r in rows]), 4),
        )


def _analysis_trace_metadata(results_path: str, dataset_path: str, sample_limit: int) -> dict[str, Any]:
    settings = get_langfuse_settings()
    return {
        "app_env": settings.app_env,
        "app_version": settings.app_version,
        "results_path": results_path,
        "dataset_path": dataset_path,
        "sample_limit": sample_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze main2 evaluation results.")
    parser.add_argument(
        "--results-path",
        default="implementations/MY_IMPLEMENTATION/main2_eval_results.json",
        help="Path to evaluation output JSON.",
    )
    parser.add_argument(
        "--dataset-path",
        default="evals/ground-truth/ground_truth_om_handbook_v3.json",
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum number of detailed item samples to print per failure cohort.",
    )
    args = parser.parse_args()

    trace_metadata = _analysis_trace_metadata(args.results_path, args.dataset_path, args.sample_limit)
    run_trace_id: str | None = None
    run_observation_id: str | None = None

    with langfuse_span(
        name="analysis_run",
        input={"results_path": args.results_path, "dataset_path": args.dataset_path, "sample_limit": args.sample_limit},
        metadata=trace_metadata,
    ) as run_span:
        safe_update_trace(name="analysis_run", metadata=trace_metadata, tags=["analysis", "evaluation"])
        run_trace_id = get_current_trace_id()
        run_observation_id = get_current_observation_id()

        with langfuse_span(name="load_results", input=trace_metadata, metadata=trace_metadata):
            results_data = _load_json(Path(args.results_path))
            dataset_data = _load_json(Path(args.dataset_path))

        dataset_items = dataset_data.get("items", [])
        if not isinstance(dataset_items, list):
            dataset_items = []
        item_meta = {item.get("id"): item for item in dataset_items if isinstance(item, dict) and item.get("id")}

        results = results_data.get("results", [])
        if not isinstance(results, list):
            raise SystemExit("Invalid evaluation output: 'results' must be a list")

        summary = results_data.get("summary", {})
        if isinstance(summary, dict) and summary:
            print("== reported summary ==")
            print(json.dumps(summary, indent=2))

        print("\nitems", len(results))

        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)

        safety_expected_counts = Counter()
        safety_match_by_expected: dict[str, list[float]] = defaultdict(list)

        search_status_counts = Counter()
        source_count_distribution = Counter()
        unknown_section_ratios: list[float] = []

        missing_required_counter = Counter()
        high_sim_low_correctness: list[dict[str, Any]] = []
        zero_source_items: list[dict[str, Any]] = []
        traceability_zero_with_sources: list[dict[str, Any]] = []
        search_errors: list[dict[str, Any]] = []
        synthesis_failures: list[dict[str, Any]] = []
        procedural_recall_zero: list[dict[str, Any]] = []
        table_failures: list[dict[str, Any]] = []

        with langfuse_span(name="compute_cohorts", input={"result_count": len(results)}, metadata=trace_metadata):
            for result in results:
                if not isinstance(result, dict):
                    continue

                item_id = result.get("id")
                meta = item_meta.get(item_id, {}) if item_id in item_meta else {}

                category = str(result.get("category") or meta.get("category") or "<unknown>")
                behavior = str(result.get("expected_behavior") or meta.get("expected_behavior") or "answer")
                expected_safety = str(meta.get("safety_level") or "<unknown>")
                question = str(result.get("question") or meta.get("question") or "")

                evaluation_map = _eval_map(result)
                answer_similarity = float(evaluation_map.get("answer_similarity", 0.0) or 0.0)
                answer_correctness = bool(evaluation_map.get("answer_correctness", False))
                trace_recall = float(evaluation_map.get("traceability_recall", 0.0) or 0.0)
                trace_precision = float(evaluation_map.get("traceability_precision", 0.0) or 0.0)

                by_category[category].append(evaluation_map)
                by_behavior[behavior].append(evaluation_map)

                if behavior == "answer":
                    by_slice["answer_items"].append(evaluation_map)
                else:
                    by_slice["refusal_items"].append(evaluation_map)

                if _is_numeric_question(question):
                    by_slice["numeric_questions"].append(evaluation_map)
                if _is_procedural_question(question):
                    by_slice["procedural_questions"].append(evaluation_map)

                safety_expected_counts[expected_safety] += 1
                safety_match_by_expected[expected_safety].append(float(bool(evaluation_map.get("safety_level_match", False))))

                search_status = str(result.get("search_status", "unknown"))
                search_status_counts[search_status] += 1

                if search_status == "error":
                    search_errors.append(
                        {
                            "id": item_id,
                            "category": category,
                            "question": question,
                            "error": result.get("search_error"),
                        }
                    )

                output = result.get("output", {})
                if not isinstance(output, dict):
                    output = {}
                sources = output.get("sources", [])
                if not isinstance(sources, list):
                    sources = []

                source_count_distribution[len(sources)] += 1

                if len(sources) == 0:
                    zero_source_items.append(
                        {
                            "id": item_id,
                            "category": category,
                            "question": question,
                            "answer_similarity": answer_similarity,
                            "answer_correctness": answer_correctness,
                            "search_status": search_status,
                        }
                    )

                if sources:
                    unknown = sum(
                        1 for source in sources if str(source.get("section_heading", "")).strip().lower() == "unknown"
                    )
                    unknown_section_ratios.append(unknown / len(sources))

                if answer_similarity >= 0.8 and not answer_correctness:
                    high_sim_low_correctness.append(
                        {
                            "id": item_id,
                            "category": category,
                            "question": question,
                            "answer_similarity": answer_similarity,
                            "source_count": len(sources),
                        }
                    )

                if len(sources) > 0 and trace_recall == 0.0 and trace_precision == 0.0:
                    traceability_zero_with_sources.append(
                        {
                            "id": item_id,
                            "category": category,
                            "question": question,
                            "source_count": len(sources),
                        }
                    )

                if category == "procedural" and trace_recall == 0.0:
                    procedural_recall_zero.append(
                        {
                            "id": item_id,
                            "question": question,
                            "answer_similarity": answer_similarity,
                            "answer_correctness": answer_correctness,
                        }
                    )

                if category == "synthesis" and not answer_correctness:
                    synthesis_failures.append(
                        {
                            "id": item_id,
                            "question": question,
                            "answer_similarity": answer_similarity,
                            "traceability_recall": trace_recall,
                        }
                    )

                if category == "table" and not answer_correctness:
                    table_failures.append(
                        {
                            "id": item_id,
                            "question": question,
                            "answer_similarity": answer_similarity,
                            "safety_match": bool(evaluation_map.get("safety_level_match", False)),
                        }
                    )

                for evaluation in result.get("evaluations", []):
                    if not isinstance(evaluation, dict):
                        continue
                    if evaluation.get("name") != "answer_correctness":
                        continue
                    for term in _missing_required_terms(str(evaluation.get("comment", ""))):
                        missing_required_counter[term] += 1

        print("\n== search status ==")
        for status, count in sorted(search_status_counts.items()):
            print(status, count)

        _print_group_metrics("by category", by_category)
        _print_group_metrics("by expected behavior", by_behavior)
        _print_group_metrics("KPI slices", by_slice)

        print("\n== safety match by expected level ==")
        for level, count in sorted(safety_expected_counts.items()):
            print(level, count, "match", round(_mean(safety_match_by_expected[level]), 4))

        print("\n== source count distribution (top 10) ==")
        for count, freq in source_count_distribution.most_common(10):
            print(count, freq)

        print("\navg unknown_section_heading ratio", round(_mean(unknown_section_ratios), 4))

        print("\n== root-cause cohorts ==")
        print("high_similarity_low_correctness", len(high_sim_low_correctness))
        print("zero_source_items", len(zero_source_items))
        print("traceability_zero_with_sources", len(traceability_zero_with_sources))
        print("procedural_recall_zero", len(procedural_recall_zero))
        print("synthesis_failures", len(synthesis_failures))
        print("table_failures", len(table_failures))
        print("search_errors", len(search_errors))

        category_summary: dict[str, dict[str, float]] = {}
        for group, rows in sorted(by_category.items()):
            category_summary[group] = {
                "answer_correctness": round(_mean([float(bool(r.get("answer_correctness", False))) for r in rows]), 4),
                "answer_similarity": round(_mean([float(r.get("answer_similarity", 0.0) or 0.0) for r in rows]), 4),
                "safety_level_match": round(_mean([float(bool(r.get("safety_level_match", False))) for r in rows]), 4),
                "traceability_recall": round(_mean([float(r.get("traceability_recall", 0.0) or 0.0) for r in rows]), 4),
                "traceability_precision": round(_mean([float(r.get("traceability_precision", 0.0) or 0.0) for r in rows]), 4),
            }

        top_missing_required = [{"term": term, "count": count} for term, count in missing_required_counter.most_common(10)]

        summary_metric_means = summary.get("metric_means", {}) if isinstance(summary, dict) else {}
        if isinstance(summary_metric_means, dict):
            for metric_name, metric_value in summary_metric_means.items():
                if isinstance(metric_value, (int, float)):
                    safe_score_trace(name=f"analysis_mean_{metric_name}", value=float(metric_value), comment="analysis summary mean")

        analysis_summary = {
            "item_count": len(results),
            "search_status_counts": dict(search_status_counts),
            "source_count_distribution": dict(source_count_distribution),
            "metric_means": summary_metric_means if isinstance(summary_metric_means, dict) else {},
            "category_summary": category_summary,
            "high_similarity_low_correctness": len(high_sim_low_correctness),
            "zero_source_items": len(zero_source_items),
            "traceability_zero_with_sources": len(traceability_zero_with_sources),
            "low_traceability_count": len(traceability_zero_with_sources),
            "low_correctness_high_similarity_count": len(high_sim_low_correctness),
            "procedural_recall_zero": len(procedural_recall_zero),
            "synthesis_failures": len(synthesis_failures),
            "table_failures": len(table_failures),
            "search_errors": len(search_errors),
            "frequent_missing_required_patterns": top_missing_required,
        }

        safe_update_trace(output=truncate_value(analysis_summary), metadata={**trace_metadata, **analysis_summary}, tags=["analysis", "summary"])
        safe_update_observation(run_span, output=truncate_value(analysis_summary), metadata=analysis_summary)

        print("\n== frequent missing-required terms (top 20) ==")
        for term, count in missing_required_counter.most_common(20):
            print(count, term)

        def print_samples(title: str, rows: list[dict[str, Any]]) -> None:
            print(f"\n== {title} (sample) ==")
            for row in rows[: max(args.sample_limit, 0)]:
                print(json.dumps(row, ensure_ascii=True))

        print_samples("high similarity but wrong", high_sim_low_correctness)
        print_samples("zero sources", zero_source_items)
        print_samples("traceability zero despite sources", traceability_zero_with_sources)
        print_samples("procedural recall zero", procedural_recall_zero)
        print_samples("synthesis failures", synthesis_failures)
        print_samples("table failures", table_failures)
        print_samples("search errors", search_errors)

    safe_flush_langfuse()


if __name__ == "__main__":
    main()
