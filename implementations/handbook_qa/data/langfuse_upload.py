"""Convert the grounded-QA ground-truth dataset and upload it to Langfuse.

The ground-truth file follows ``evals/ground-truth/schema.json``. Each item is
richer than the ``input`` / ``expected_output`` shape that Langfuse datasets use,
so this script flattens every item into a Langfuse-compatible record:

- ``input``            <- ``question``
- ``expected_output``  <- ``expected_answer.text`` (the canonical reference answer)
- ``metadata``         <- everything the graders need to score the three axes
                           (safety level, traceability sources, answer-match
                           config, acceptable variants, must/must-not include).

It then reuses the shared :func:`upload_dataset_to_langfuse` utility for
progress tracking and deterministic, de-duplicated item IDs.

Usage:
    # Upload the illustrative example dataset (default)
    python langfuse_upload.py

    # Upload a real ground-truth file under a custom dataset name
    python langfuse_upload.py \
        --ground-truth-path ../../../evals/ground-truth/handbook_v1.json \
        --dataset-name "OMHandbook-QA-v1"
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Resolve paths relative to this file so the script works regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
DEFAULT_GROUND_TRUTH_PATH = _REPO_ROOT / "evals" / "ground-truth" / "ground_truth_om_handbook_v3.json"
DEFAULT_DATASET_NAME = "OMHandbook-QA"


def _load_ground_truth(path: Path) -> dict[str, Any]:
    """Load and minimally validate a ground-truth file.

    Parameters
    ----------
    path : Path
        Path to a JSON file conforming to ``evals/ground-truth/schema.json``.

    Returns
    -------
    dict[str, Any]
        Parsed ground-truth document with at least ``items``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file does not contain a non-empty ``items`` list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    document = json.loads(path.read_text(encoding="utf-8"))

    items = document.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Ground-truth file '{path}' has no 'items' list.")

    return document


def _convert_item(item: dict[str, Any], default_answer_match: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten one schema item into a Langfuse dataset record.

    Parameters
    ----------
    item : dict[str, Any]
        A single ground-truth item from the schema's ``items`` array.
    default_answer_match : dict[str, Any] | None
        Dataset-level default answer-match config applied when the item omits
        its own ``answer_match`` override.

    Returns
    -------
    dict[str, Any]
        A record with ``id``, ``input``, ``expected_output`` and ``metadata``
        keys understood by :func:`upload_dataset_to_langfuse`.
    """
    expected = item.get("expected_answer", {})

    # Per-item answer-match overrides the dataset default; either may be absent.
    answer_match = item.get("answer_match", default_answer_match)

    metadata: dict[str, Any] = {
        "id": item["id"],
        "category": item.get("category"),
        "expected_behavior": item.get("expected_behavior", "answer"),
        "safety_level": item["safety_level"],
        # Refusal / out-of-scope items may legitimately omit traceability.
        "traceability": item.get("traceability", []),
        "tags": item.get("tags", []),
        "acceptable_variants": expected.get("acceptable_variants", []),
        "must_include": expected.get("must_include", []),
        "must_not_include": expected.get("must_not_include", []),
    }
    if answer_match is not None:
        metadata["answer_match"] = answer_match

    return {
        "id": item["id"],
        "input": item["question"],
        "expected_output": expected["text"],
        "metadata": metadata,
    }


async def upload_ground_truth_to_langfuse(ground_truth_path: Path, dataset_name: str) -> None:
    """Convert a ground-truth file and upload it to Langfuse.

    Parameters
    ----------
    ground_truth_path : Path
        Path to the ground-truth JSON file.
    dataset_name : str
        Name for the dataset in Langfuse.
    """
    document = _load_ground_truth(ground_truth_path)
    items: list[dict[str, Any]] = document["items"]
    default_answer_match = document.get("defaults", {}).get("answer_match")

    logger.info("Converting %d ground-truth item(s) from '%s'", len(items), ground_truth_path)
    records = [_convert_item(item, default_answer_match) for item in items]

    # Write to a temporary JSONL file consumed by the shared upload utility,
    # which handles progress tracking and deterministic de-duplication IDs.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix=f"handbook_qa_{dataset_name}_",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        for record in records:
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        await upload_dataset_to_langfuse(dataset_path=str(temp_path), dataset_name=dataset_name)
    finally:
        if temp_path.exists():
            temp_path.unlink()
            logger.debug("Removed temporary file: %s", temp_path)


@click.command()
@click.option(
    "--ground-truth-path",
    default=str(DEFAULT_GROUND_TRUTH_PATH),
    type=click.Path(path_type=Path),
    help="Path to the ground-truth JSON file (schema: evals/ground-truth/schema.json).",
)
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name for the dataset in Langfuse.",
)
def cli(ground_truth_path: Path, dataset_name: str) -> None:
    """Upload the grounded-QA ground-truth dataset to Langfuse."""
    asyncio.run(upload_ground_truth_to_langfuse(ground_truth_path, dataset_name))


if __name__ == "__main__":
    cli()
