#!/usr/bin/env python3
r"""Create a Vertex AI Search data store from the O&M Handbook chunks.

Provisions the GCS bucket, transforms and uploads the handbook chunks JSONL,
creates a CONTENT_REQUIRED structured data store, imports the documents, and
waits for indexing.

Usage
-----
    # Authenticate first (or rely on the GCE service account in CI)
    gcloud auth application-default login

    # Run from the repo root
    uv run python -m scripts.create_handbook_datastore \\
        --bucket <globally-unique-bucket-name> \\
        [--project agentic-ai-evaluation-bootcamp] \\
        [--datastore-id om-handbook] \\
        [--input data/processed/om_handbook/chunks.jsonl]

After the script finishes it prints the VERTEX_AI_DATASTORE_ID value
to add to your .env file.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests


DISCOVERY_ENGINE_BASE = "https://discoveryengine.googleapis.com/v1"
STORAGE_BASE = "https://storage.googleapis.com/storage/v1"
STORAGE_UPLOAD_BASE = "https://storage.googleapis.com/upload/storage/v1"

# Default handbook chunks file relative to the repo root
DEFAULT_INPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "om_handbook" / "chunks.jsonl"

# Scalar chunk fields preserved as searchable/filterable metadata in structData.
# Bulky or nested fields (content, tables, metadata, byte offsets) are excluded.
STRUCT_DATA_FIELDS = (
    "title",
    "heading_text",
    "section_id",
    "parent_section_id",
    "section_number",
    "level",
    "page_start",
    "page_end",
)


def get_session() -> google.auth.transport.requests.AuthorizedSession:
    """Return an authorised requests session using Application Default Credentials."""
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return google.auth.transport.requests.AuthorizedSession(credentials)


def create_bucket(session, project: str, bucket: str) -> None:
    """Create a GCS bucket in us-central1, skipping if it already exists."""
    url = f"{STORAGE_BASE}/b?project={project}"
    body = {"name": bucket, "location": "us-central1", "storageClass": "STANDARD"}
    resp = session.post(url, json=body)
    if resp.status_code == 409:
        print(f"  Bucket gs://{bucket} already exists — skipping creation.")
    elif resp.status_code in (200, 201):
        print(f"  Created bucket gs://{bucket}")
    else:
        print(f"  Error creating bucket: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()


def transform_chunks(source_path: Path) -> bytes:
    """Transform O&M Handbook chunks JSONL to Discovery Engine CONTENT_REQUIRED format.

    Handbook chunk format (flat):
        {"chunk_id": "x", "content": "...", "title": "...", "section_id": "...", ...}

    Discovery Engine CONTENT_REQUIRED format:
        {
            "id": "x",
            "content": {"mimeType": "text/plain", "rawBytes": "<base64>"},
            "structData": {...}
        }

    The ``content`` field becomes the indexed document content (stored as base64
    rawBytes). Selected scalar fields (see ``STRUCT_DATA_FIELDS``) become
    metadata in ``structData``; ``None`` values are dropped.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {source_path}")

    output_lines = []
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").strip().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)

        doc_id = row.get("chunk_id")
        if not doc_id:
            raise ValueError(f"Missing 'chunk_id' on line {line_number} of {source_path}")

        text = row.get("content", "") or ""
        struct_data = {field: row[field] for field in STRUCT_DATA_FIELDS if row.get(field) is not None}

        doc = {
            "id": doc_id,
            "content": {
                "mimeType": "text/plain",
                "rawBytes": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            },
            "structData": struct_data,
        }
        output_lines.append(json.dumps(doc))

    if not output_lines:
        raise ValueError(f"No chunks found in {source_path}")

    print(f"  Transformed {len(output_lines)} chunk(s) from {source_path.name}")
    return "\n".join(output_lines).encode("utf-8")


def upload_chunks(session, bucket: str, object_name: str, input_path: Path) -> None:
    """Transform and upload the handbook chunks to GCS in CONTENT_REQUIRED format."""
    payload = transform_chunks(input_path)
    url = f"{STORAGE_UPLOAD_BASE}/b/{bucket}/o?uploadType=media&name={object_name}"
    resp = session.post(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    print(f"  Uploaded {input_path.name} → gs://{bucket}/{object_name}")


def create_datastore(session, project: str, datastore_id: str) -> None:
    """Create a CONTENT_REQUIRED structured search data store, skipping if it exists."""
    url = (
        f"{DISCOVERY_ENGINE_BASE}/projects/{project}/locations/global"
        f"/collections/default_collection/dataStores?dataStoreId={datastore_id}"
    )
    body = {
        "displayName": "O&M Handbook",
        "industryVertical": "GENERIC",
        "contentConfig": "CONTENT_REQUIRED",
        "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
    }
    resp = session.post(url, json=body)
    if resp.status_code == 409:
        print(f"  Data store '{datastore_id}' already exists — skipping creation.")
    elif resp.status_code in (200, 201):
        print(f"  Created data store '{datastore_id}'")
        # Allow a moment for the data store to become fully ready
        time.sleep(5)
    else:
        print(f"  Error creating data store: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()


def import_documents(session, project: str, datastore_id: str, gcs_uri: str) -> str:
    """Trigger an async document import from GCS. Returns the operation name."""
    url = (
        f"{DISCOVERY_ENGINE_BASE}/projects/{project}/locations/global"
        f"/collections/default_collection/dataStores/{datastore_id}"
        f"/branches/default_branch/documents:import"
    )
    body = {
        "gcsSource": {
            "inputUris": [gcs_uri],
            # "document" matches our JSONL format:
            # {id, content:{mimeType,rawBytes}, structData:{...}}
            "dataSchema": "document",
        },
        # FULL replaces all existing documents, keeping the store deterministic
        "reconciliationMode": "FULL",
    }
    resp = session.post(url, json=body)
    resp.raise_for_status()
    operation_name = resp.json()["name"]
    print(f"  Import operation started: {operation_name}")
    return operation_name


def wait_for_operation(
    session,
    operation_name: str,
    timeout_sec: int = 600,
    poll_interval: int = 15,
) -> dict:
    """Poll the operation until it is done or the timeout is reached."""
    url = f"{DISCOVERY_ENGINE_BASE}/{operation_name}"
    start = time.time()
    deadline = start + timeout_sec

    while time.time() < deadline:
        resp = session.get(url)
        resp.raise_for_status()
        op = resp.json()

        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Import operation failed: {op['error']}")
            # Check per-document failure count in metadata
            metadata = op.get("metadata", {})
            failure_count = int(metadata.get("failureCount", 0))
            total_count = int(metadata.get("totalCount", 0))
            if failure_count > 0:
                samples = op.get("response", {}).get("errorSamples", [])
                sample_msg = samples[0]["message"] if samples else "unknown error"
                raise RuntimeError(
                    f"Import completed but {failure_count}/{total_count} documents failed. First error: {sample_msg}"
                )
            print(f"  Indexing complete — {total_count} documents imported.")
            return op

        elapsed = int(time.time() - start)
        print(f"  Indexing in progress… ({elapsed}s elapsed, checking again in {poll_interval}s)")
        time.sleep(poll_interval)

    raise TimeoutError(f"Operation did not complete within {timeout_sec}s: {operation_name}")


def main() -> None:
    """Parse CLI arguments and provision the O&M Handbook Vertex AI Search data store."""
    parser = argparse.ArgumentParser(
        description="Provision a Vertex AI Search data store from the O&M Handbook chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        default="agentic-ai-evaluation-bootcamp",
        help="GCP project ID (default: agentic-ai-evaluation-bootcamp)",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name for staging the import file (must be globally unique)",
    )
    parser.add_argument(
        "--datastore-id",
        default="om-handbook",
        help="Vertex AI Search data store ID (default: om-handbook)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the chunks JSONL file (default: {DEFAULT_INPUT_PATH})",
    )
    args = parser.parse_args()

    gcs_object = "om-handbook/chunks.jsonl"
    gcs_uri = f"gs://{args.bucket}/{gcs_object}"
    datastore_resource = (
        f"projects/{args.project}/locations/global/collections/default_collection/dataStores/{args.datastore_id}"
    )

    print("Vertex AI Search — O&M Handbook data store provisioning")
    print("=" * 55)
    print(f"  Project:    {args.project}")
    print(f"  Bucket:     gs://{args.bucket}")
    print(f"  Data store: {datastore_resource}")
    print(f"  Input:      {args.input}")
    print()

    session = get_session()

    print("Step 1/5  Creating GCS bucket…")
    create_bucket(session, args.project, args.bucket)

    print("Step 2/5  Uploading handbook chunks to GCS…")
    upload_chunks(session, args.bucket, gcs_object, args.input)

    print("Step 3/5  Creating Vertex AI Search data store…")
    create_datastore(session, args.project, args.datastore_id)

    print("Step 4/5  Importing documents…")
    operation_name = import_documents(session, args.project, args.datastore_id, gcs_uri)

    print("Step 5/5  Waiting for indexing (may take several minutes)…")
    wait_for_operation(session, operation_name)

    print()
    print("=" * 55)
    print("Done! Add this to your .env file:")
    print()
    print(f'VERTEX_AI_DATASTORE_ID="{datastore_resource}"')
    print("=" * 55)


if __name__ == "__main__":
    main()
