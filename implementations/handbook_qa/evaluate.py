"""Evaluate the Handbook QA agent using Langfuse experiments.

Runs the grounded-QA agent against a Langfuse dataset (uploaded by
``data/langfuse_upload.py``) and scores each answer on the three ground-truth
axes via the evaluators in :mod:`evaluators`:

- ``answer_correctness`` / ``answer_similarity``
- ``safety_level_match``
- ``traceability_recall`` / ``traceability_precision`` / ``traceability_complete``

Scores are logged to Langfuse for analysis and run-to-run comparison.

The agent itself lives in ``implementations/handbook_qa/agent.py`` (a
prerequisite, grounded on the Vertex AI Search data store). The task adapter
below expects the agent's response to expose ``text`` and, ideally,
``safety_level`` and ``sources`` so the safety and traceability axes can be
scored. Until those structured fields exist, those evaluators report a zero
score with an explanatory comment rather than failing the run.

Usage:
    python -m implementations.handbook_qa.evaluate
    python -m implementations.handbook_qa.evaluate --user-id calen
    python -m implementations.handbook_qa.evaluate \
        --dataset-name "OMHandbook-QA-v1" --experiment-name "v1-baseline" --user-id sumit
"""

import asyncio
import getpass
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.evaluation.types import ExperimentResult
from dotenv import load_dotenv

from implementations.handbook_qa.evaluators import build_evaluators


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "OMHandbook-QA"

#: Prefix identifying this evaluation for all runs (keeps repeats grouped).
EXPERIMENT_PREFIX = "handbook_qa"


def build_experiment_name(dataset_name: str, timestamp: datetime | None = None) -> str:
    """Return a standardized, sortable experiment name for a single run.

    Format: ``handbook_qa-<dataset>-<YYYYMMDDTHHMMSSZ>`` (UTC). The constant
    prefix and dataset keep repeated runs of the same evaluation grouped and
    sorted together, while the timestamp makes every run unique so the same
    evaluation can be re-run without name collisions.

    Parameters
    ----------
    dataset_name : str
        The Langfuse dataset being evaluated.
    timestamp : datetime, optional
        Run time; defaults to the current UTC time.

    Returns
    -------
    str
        The standardized experiment name.
    """
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{EXPERIMENT_PREFIX}-{dataset_name}-{ts}"


def _build_structured_output(response: Any) -> dict[str, Any]:
    """Normalize an agent response into the evaluator output contract.

    Reads ``text`` and the optional ``safety_level`` / ``sources`` fields from
    the agent response, tolerating attributes or dict keys.

    Parameters
    ----------
    response : Any
        The object returned by the handbook agent.

    Returns
    -------
    dict[str, Any]
        ``{"text": str, "safety_level": str | None, "sources": list,
        "retrievals": list, "trace": list}``.
    """

    def _get(field: str, default: Any) -> Any:
        if isinstance(response, dict):
            return response.get(field, default)
        return getattr(response, field, default)

    return {
        "text": str(_get("text", response)),
        "safety_level": _get("safety_level", None),
        "sources": _get("sources", []) or [],
        "retrievals": _get("retrievals", []) or [],
        "trace": _get("trace", []) or [],
    }


def make_agent_task(session_id: str, user_id: str) -> Any:
    """Build the experiment task bound to a shared Langfuse session and user.

    All items in a single experiment run are tagged with the same
    ``session_id`` so their traces are grouped under one Langfuse session, and
    the same ``user_id`` so the run is attributed to whoever submitted it.

    Parameters
    ----------
    session_id : str
        Session identifier shared by every dataset item in this run.
    user_id : str
        Langfuse user identifier (the submitter, e.g. "calen") applied to every
        trace in this run.

    Returns
    -------
    Any
        An async task callable compatible with ``run_experiment``.
    """

    async def agent_task(*, item: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
        """Run the Handbook QA agent on a single dataset item.

        Parameters
        ----------
        item : Any
            The Langfuse experiment item; ``item.input`` is the question.

        Returns
        -------
        dict[str, Any]
            Structured output consumed by the evaluators (text, safety_level, sources).
        """
        # Imported lazily so the dataset upload and evaluator unit tests do not
        # require the (datastore-dependent) agent to be importable yet.
        from implementations.handbook_qa.agent import HandbookGroundedAgent  # noqa: PLC0415

        question = item.input
        logger.info("Running agent on: %s...", question[:80])

        # Group this item's trace into the run-wide session and attribute it to
        # the submitter so Langfuse can track who ran what.
        client_manager = AsyncClientManager.get_instance()
        client_manager.langfuse_client.update_current_trace(
            session_id=session_id,
            user_id=user_id,
        )

        agent = HandbookGroundedAgent()
        response = await agent.answer_async(question)

        structured = _build_structured_output(response)

        # Attach the rich response to the span metadata so it is inspectable in
        # Langfuse without cluttering the scored output.
        client_manager.langfuse_client.update_current_span(metadata=structured)

        return structured

    return agent_task


def score_session(session_id: str, result: ExperimentResult) -> list[dict[str, Any]]:
    """Aggregate item-level evaluations into session-level Langfuse scores.

    Each per-item metric is rolled up across the whole run and attached to the
    run's session via ``create_score(session_id=...)``:

    - BOOLEAN metrics (e.g. ``answer_correctness``) become a pass *rate*
      (fraction of items that passed), published as ``session/<name>_rate``.
    - NUMERIC metrics (e.g. ``answer_similarity``, ``traceability_recall``)
      become a *mean*, published as ``session/<name>_mean``.

    Parameters
    ----------
    session_id : str
        The session grouping all item traces for this run.
    result : ExperimentResult
        The completed experiment result holding per-item evaluations.

    Returns
    -------
    list[dict[str, Any]]
        The session scores that were published (for logging/inspection).
    """
    client = AsyncClientManager.get_instance().langfuse_client

    values_by_name: dict[str, list[Any]] = defaultdict(list)
    data_type_by_name: dict[str, str | None] = {}
    for item_result in result.item_results:
        for evaluation in item_result.evaluations:
            values_by_name[evaluation.name].append(evaluation.value)
            data_type_by_name.setdefault(evaluation.name, evaluation.data_type)

    published: list[dict[str, Any]] = []
    for name, values in values_by_name.items():
        # bool is a subclass of int, so this also keeps boolean values.
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if not numeric:
            continue

        if data_type_by_name.get(name) == "BOOLEAN":
            passed = sum(1 for v in values if bool(v))
            score_name = f"session/{name}_rate"
            score_value = passed / len(values)
            comment = f"{passed}/{len(values)} items passed"
        else:
            score_name = f"session/{name}_mean"
            score_value = sum(numeric) / len(numeric)
            comment = f"mean over {len(numeric)} item(s)"

        client.create_score(
            session_id=session_id,
            name=score_name,
            value=score_value,
            data_type="NUMERIC",
            comment=comment,
        )
        published.append({"name": score_name, "value": score_value, "comment": comment})

    # Always record how many items the session covers.
    client.create_score(
        session_id=session_id,
        name="session/item_count",
        value=float(len(result.item_results)),
        data_type="NUMERIC",
    )
    client.flush()
    return published


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    user_id: str,
    max_concurrency: int = 1,
    limit: int | None = None,
) -> ExperimentResult:
    """Run the handbook QA evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    user_id : str
        Submitter attributed to every trace in the run (e.g. "jane").
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    limit : int | None, optional
        If set, evaluate only the first ``limit`` dataset items (useful for
        quick smoke tests). By default all items are used.
    """
    client_manager = AsyncClientManager.get_instance()

    # One session per experiment run groups all dataset-item traces together.
    # The experiment name already carries a UTC timestamp; a short uuid guards
    # against collisions when the same evaluation is re-run within a second.
    session_id = f"{experiment_name}-{uuid.uuid4().hex[:8]}"

    # Which evaluators actually run is driven by eval_config.yaml.
    evaluators = build_evaluators()
    logger.info(
        "Active evaluators (%d): %s",
        len(evaluators),
        ", ".join(getattr(fn, "__name__", str(fn)) for fn in evaluators),
    )

    try:
        logger.info("Starting experiment '%s' on dataset '%s'", experiment_name, dataset_name)
        logger.info("Langfuse session: %s (user: %s)", session_id, user_id)
        if limit is not None:
            # Slice the dataset to the first `limit` items and run via the
            # client-level API. Passing Langfuse DatasetItem objects keeps the
            # run linked to the dataset in the Langfuse UI.
            dataset = client_manager.langfuse_client.get_dataset(dataset_name)
            items = list(dataset.items)[:limit]
            logger.info("Limiting evaluation to %d of %d dataset item(s)", len(items), len(dataset.items))
            result = client_manager.langfuse_client.run_experiment(
                name=experiment_name,
                description="Handbook QA: answer correctness, safety level, and traceability",
                data=items,
                task=make_agent_task(session_id, user_id),
                evaluators=evaluators,
                max_concurrency=max_concurrency,
            )
        else:
            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Handbook QA: answer correctness, safety level, and traceability",
                task=make_agent_task(session_id, user_id),
                evaluators=evaluators,
                max_concurrency=max_concurrency,
            )
        logger.info("Experiment complete: %s", result)

        # Roll item-level evaluations up into session-level scores.
        session_scores = score_session(session_id, result)
        logger.info("Published %d session-level score(s) to '%s'", len(session_scores), session_id)
        for score in session_scores:
            logger.info("  %s = %.4f (%s)", score["name"], score["value"], score["comment"])

        return result
    finally:
        logger.info("Closing client manager and flushing data...")
        try:
            await client_manager.close()
            await asyncio.sleep(0.1)
        except Exception as exc:
            logger.warning("Cleanup warning: %s", exc)


@click.command()
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, help="Langfuse dataset to evaluate against.")
@click.option(
    "--experiment-name",
    default=None,
    help="Name for this run. Defaults to the standardized 'handbook_qa-<dataset>-<UTC timestamp>' format.",
)
@click.option(
    "--user-id",
    default=getpass.getuser(),
    help="Submitter attributed to every trace in the run (e.g. 'calen'). Defaults to the OS user.",
)
@click.option("--max-concurrency", default=1, type=int, help="Maximum concurrent agent runs (default: 1).")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Evaluate only the first N dataset items (e.g. --limit 10 for a quick smoke test).",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an eval-selection YAML (defaults to implementations/handbook_qa/eval_config.yaml).",
)
def cli(
    dataset_name: str,
    experiment_name: str | None,
    user_id: str,
    max_concurrency: int,
    limit: int | None,
    config_path: str | None,
) -> None:
    """Run Handbook QA evaluation using Langfuse experiments."""
    if config_path:
        os.environ["HANDBOOK_EVAL_CONFIG"] = config_path
    experiment_name = experiment_name or build_experiment_name(dataset_name)
    asyncio.run(run_evaluation(dataset_name, experiment_name, user_id, max_concurrency, limit))


if __name__ == "__main__":
    cli()
