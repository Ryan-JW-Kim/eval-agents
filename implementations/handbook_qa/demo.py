"""Local UI for exploring the Handbook QA agent and its evaluation.

A small Gradio app that lets you:

1. Pick a question from the ground-truth dataset
   (``evals/ground-truth/ground_truth_om_handbook_v3.json``).
2. See the expected answer, safety level and required grounding sources.
3. Run the grounded agent on that question and inspect its answer, classified
   safety level and cited sources.
4. Score the answer with the same item-level evaluators used by the offline
   experiment (``answer_correctness`` / ``answer_similarity``,
   ``safety_level_match``, ``traceability_*``).

This runs the agent and the graders directly (no Langfuse round-trip), so it is
handy for quickly eyeballing behaviour on individual items.

Usage:
    python -m implementations.handbook_qa.demo
    python -m implementations.handbook_qa.demo --ground-truth-path <path> --share
"""

import logging
from pathlib import Path
from typing import Any

import click
import gradio as gr
from dotenv import load_dotenv
from langfuse.experiment import Evaluation

from implementations.handbook_qa.data.langfuse_upload import (
    DEFAULT_GROUND_TRUTH_PATH,
    _convert_item,
    _load_ground_truth,
)
from implementations.handbook_qa.evaluators import (
    answer_completeness_evaluator,
    answer_correctness_evaluator,
    answer_correctness_judge_evaluator,
    answer_relevance_evaluator,
    citation_count_evaluator,
    citation_presence_evaluator,
    efficiency_evaluator,
    evidence_grounded_reasoning_evaluator,
    groundedness_evaluator,
    keyword_constraints_evaluator,
    query_quality_evaluator,
    reasoning_coherence_evaluator,
    refusal_appropriateness_evaluator,
    safety_awareness_evaluator,
    safety_justification_evaluator,
    safety_level_evaluator,
    safety_level_valid_evaluator,
    safety_underrated_evaluator,
    tool_selection_evaluator,
    traceability_evaluator,
)


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


#: Lazily-created singleton agent so we build the (datastore-backed) grounding
#: tool only once and reuse it across runs.
_AGENT: Any = None


def _get_agent() -> Any:
    """Return a cached :class:`HandbookGroundedAgent`, creating it on first use.

    The import and instantiation are deferred so the app can start (and show a
    clear error in the UI) even if the agent's datastore config is missing.
    """
    global _AGENT
    if _AGENT is None:
        from implementations.handbook_qa.agent import HandbookGroundedAgent  # noqa: PLC0415

        _AGENT = HandbookGroundedAgent()
    return _AGENT


def _load_records(ground_truth_path: Path) -> dict[str, dict[str, Any]]:
    """Load ground-truth items and flatten them into evaluator-ready records.

    Reuses :func:`_convert_item` from the Langfuse uploader so the records here
    carry exactly the ``input`` / ``expected_output`` / ``metadata`` shape the
    graders expect, keeping the UI consistent with the offline experiment.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of item id to its flattened record.
    """
    document = _load_ground_truth(ground_truth_path)
    default_answer_match = document.get("defaults", {}).get("answer_match")
    records: dict[str, dict[str, Any]] = {}
    for item in document["items"]:
        record = _convert_item(item, default_answer_match)
        records[record["id"]] = record
    logger.info("Loaded %d ground-truth item(s) from '%s'", len(records), ground_truth_path)
    return records


def _dropdown_choices(records: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """Build ``(label, id)`` choices for the question dropdown."""
    choices: list[tuple[str, str]] = []
    for item_id, record in records.items():
        question = record["input"]
        label = f"{item_id} — {question[:90]}{'…' if len(question) > 90 else ''}"
        choices.append((label, item_id))
    return choices


def _format_value(value: Any) -> str:
    """Render an evaluation value for the scores table."""
    if isinstance(value, bool):
        return "✅ pass" if value else "❌ fail"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _run_evaluators(output: dict[str, Any], record: dict[str, Any]) -> list[Evaluation]:
    """Score an agent ``output`` against a ground-truth ``record``.

    Runs the same item-level graders as the offline experiment -- the three core
    axes plus the ported heuristic and LLM-judge batteries -- and returns a flat
    list of :class:`Evaluation` results. Judge evaluators degrade to nothing when
    the judge is unavailable or the item is out of scope.
    """
    metadata = record["metadata"]
    expected_output = record["expected_output"]
    question = record["input"]

    def _many(result: Evaluation | list[Evaluation]) -> list[Evaluation]:
        return result if isinstance(result, list) else [result]

    results: list[Evaluation] = []
    # Core three-axis evaluators.
    results.extend(
        answer_correctness_evaluator(
            output=output, expected_output=expected_output, metadata=metadata
        )
    )
    results.append(safety_level_evaluator(output=output, metadata=metadata))
    results.extend(traceability_evaluator(output=output, metadata=metadata))
    # Heuristic battery.
    results.append(safety_level_valid_evaluator(output=output, metadata=metadata))
    results.append(safety_underrated_evaluator(output=output, metadata=metadata))
    results.append(citation_presence_evaluator(output=output, metadata=metadata))
    results.append(citation_count_evaluator(output=output, metadata=metadata))
    results.extend(keyword_constraints_evaluator(output=output, metadata=metadata))
    # LLM-judge (output + trace). Each returns [] when unavailable / out of scope.
    for judge in (
        answer_correctness_judge_evaluator,
        answer_relevance_evaluator,
        answer_completeness_evaluator,
        groundedness_evaluator,
        safety_justification_evaluator,
        refusal_appropriateness_evaluator,
        reasoning_coherence_evaluator,
        tool_selection_evaluator,
        query_quality_evaluator,
        evidence_grounded_reasoning_evaluator,
        efficiency_evaluator,
        safety_awareness_evaluator,
    ):
        results.extend(
            _many(
                judge(
                    input=question,
                    output=output,
                    expected_output=expected_output,
                    metadata=metadata,
                )
            )
        )
    return results


def _on_select(item_id: str, records: dict[str, dict[str, Any]]) -> tuple[str, str, str, str, Any]:
    """Populate the expected-answer panel when a question is selected."""
    if not item_id or item_id not in records:
        return "", "", "", "", []
    record = records[item_id]
    metadata = record["metadata"]
    category = metadata.get("category") or "—"
    behavior = metadata.get("expected_behavior") or "—"
    context = f"**Category:** {category}  |  **Expected behavior:** {behavior}"
    return (
        record["input"],
        context,
        record["expected_output"],
        metadata.get("safety_level") or "—",
        metadata.get("traceability", []),
    )


async def _on_run(
    item_id: str, records: dict[str, dict[str, Any]]
) -> tuple[str, str, Any, list[str], list[list[str]]]:
    """Run the agent on the selected question and score its answer.

    Returns
    -------
    tuple
        Agent answer text, classified safety level, cited sources, the search
        queries the agent issued, and the evaluation-scores table rows.
    """
    if not item_id or item_id not in records:
        return "Select a question first.", "—", [], [], []

    record = records[item_id]
    question = record["input"]

    try:
        agent = _get_agent()
        response = await agent.answer_async(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run failed")
        return f"⚠️ Agent run failed: {exc}", "—", [], [], []

    output = {
        "text": response.text,
        "safety_level": response.safety_level,
        "sources": response.sources,
        "retrievals": response.retrievals,
        "trace": response.trace,
    }

    try:
        evaluations = _run_evaluators(output, record)
        rows = [[e.name, _format_value(e.value), e.comment or ""] for e in evaluations]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed")
        rows = [["error", "—", f"evaluation failed: {exc}"]]

    return (
        response.text,
        response.safety_level or "(not classified)",
        response.sources,
        response.search_queries,
        rows,
    )


def build_demo(records: dict[str, dict[str, Any]]) -> gr.Blocks:
    """Construct the Gradio ``Blocks`` app for the given ground-truth records."""
    choices = _dropdown_choices(records)
    first_id = choices[0][1] if choices else None
    init_question, init_context, init_answer, init_safety, init_sources = (
        _on_select(first_id, records) if first_id else ("", "", "", "", [])
    )

    with gr.Blocks(title="Handbook QA — Agent & Evaluation Explorer") as demo:
        records_state = gr.State(records)

        gr.Markdown(
            "# Handbook QA — Agent & Evaluation Explorer\n"
            "Pick a ground-truth question, run the grounded agent, and compare "
            "its answer, safety level and cited sources against the expected "
            "values with the offline graders."
        )

        with gr.Row():
            # Left: question + expected ground truth.
            with gr.Column(scale=1):
                question_dd = gr.Dropdown(
                    choices=choices,
                    value=first_id,
                    label="Ground-truth question",
                    filterable=True,
                )
                question_box = gr.Textbox(
                    value=init_question, label="Question", lines=3, interactive=False
                )
                context_md = gr.Markdown(init_context)
                run_btn = gr.Button("Run agent", variant="primary")

                gr.Markdown("### Expected")
                expected_answer = gr.Textbox(
                    value=init_answer, label="Expected answer", lines=5, interactive=False
                )
                expected_safety = gr.Textbox(
                    value=init_safety, label="Expected safety level", interactive=False
                )
                expected_sources = gr.JSON(
                    value=init_sources, label="Required grounding sources"
                )

            # Right: agent output + evaluation.
            with gr.Column(scale=1):
                gr.Markdown("### Agent output")
                agent_answer = gr.Textbox(label="Answer", lines=6, interactive=False)
                agent_safety = gr.Textbox(label="Classified safety level", interactive=False)
                agent_sources = gr.JSON(label="Cited sources")
                agent_queries = gr.JSON(label="Search queries issued")

                gr.Markdown("### Evaluation")
                scores_df = gr.Dataframe(
                    headers=["metric", "value", "comment"],
                    datatype=["str", "str", "str"],
                    label="Scores",
                    wrap=True,
                    interactive=False,
                )

        question_dd.change(
            fn=_on_select,
            inputs=[question_dd, records_state],
            outputs=[question_box, context_md, expected_answer, expected_safety, expected_sources],
        )
        run_btn.click(
            fn=_on_run,
            inputs=[question_dd, records_state],
            outputs=[agent_answer, agent_safety, agent_sources, agent_queries, scores_df],
        )

    return demo


@click.command()
@click.option(
    "--ground-truth-path",
    default=str(DEFAULT_GROUND_TRUTH_PATH),
    type=click.Path(path_type=Path),
    help="Path to the ground-truth JSON file (schema: evals/ground-truth).",
)
@click.option("--share", is_flag=True, default=False, help="Expose the app via a public Gradio link.")
def cli(ground_truth_path: Path, share: bool) -> None:
    """Launch the Handbook QA agent + evaluation explorer UI."""
    records = _load_records(ground_truth_path)
    demo = build_demo(records)
    demo.launch(share=share)


if __name__ == "__main__":
    cli()
