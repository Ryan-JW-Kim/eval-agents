#!/usr/bin/env python3
r"""Ask a question against the Hitachi O&M handbook Vertex AI Search data store.

Sends a search query to the data store's ``default_search`` serving config and
prints the full raw JSON response so you can inspect exactly what Vertex AI
Search returns (results, summary, attribution token, metadata, etc.).

Authentication uses the attached GCE service account by default (which has the
required Discovery Engine permissions), falling back to Application Default
Credentials if no metadata server is reachable.

Usage
-----
    # Run from the repo root
    uv run python -m scripts.query_handbook_datastore \\
        "What is the inspection frequency for relief valves?"

    # Override defaults
    uv run python -m scripts.query_handbook_datastore \\
        --project agentic-ai-evaluation-bootcamp \\
        --location global \\
        --datastore-id hitachi-om-pdf-1_1782840198465 \\
        --page-size 5 \\
        "How often must fire extinguishers be inspected?"
"""

import argparse
import json
import sys

import google.auth
import google.auth.transport.requests
from google.auth import compute_engine
from google.genai import Client, types


DISCOVERY_ENGINE_BASE = "https://discoveryengine.googleapis.com/v1"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

DEFAULT_PROJECT = "agentic-ai-evaluation-bootcamp"
DEFAULT_LOCATION = "global"
DEFAULT_DATASTORE_ID = "hitachi-vector-bootcamp"

# Compute region for the Gemini grounded-generation call. This is the *model*
# region and may differ from the data store's ``global`` location.
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"


def get_session() -> google.auth.transport.requests.AuthorizedSession:
    """Return an authorised session, preferring the attached service account.

    The workspace's user ADC (from ``gcloud auth application-default login``)
    lacks the ``serviceusage.services.use`` permission needed by the Discovery
    Engine API, so we try the GCE metadata service account first and only fall
    back to ADC if the metadata server is unavailable.
    """
    request = google.auth.transport.requests.Request()
    try:
        creds = compute_engine.Credentials(scopes=SCOPES)
        creds.refresh(request)
        print("Auth: using attached GCE service account.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - fall back to ADC
        print(f"Auth: GCE service account unavailable ({exc}); falling back to ADC.", file=sys.stderr)
        creds, _ = google.auth.default(scopes=SCOPES)
    return google.auth.transport.requests.AuthorizedSession(creds)


def search(
    session: google.auth.transport.requests.AuthorizedSession,
    project: str,
    location: str,
    datastore_id: str,
    query: str,
    page_size: int,
    result_mode: str = "documents",
    num_neighbor_chunks: int = 0,
) -> dict:
    """Send a search request and return the parsed JSON response.

    ``result_mode`` controls the shape of the returned results:

    * ``"documents"`` (default) returns document-level metadata only
      (``structData`` with page/section, no body text).
    * ``"chunks"`` returns the actual indexed chunk text in
      ``chunk.content`` along with the chunk id and ``structData``. Use
      ``num_neighbor_chunks`` to also pull adjacent chunks for extra context.
    """
    serving_config = (
        f"projects/{project}/locations/{location}/collections/default_collection"
        f"/dataStores/{datastore_id}/servingConfigs/default_search"
    )
    url = f"{DISCOVERY_ENGINE_BASE}/{serving_config}:search"
    if result_mode == "chunks":
        # CHUNKS mode returns chunk.content (the actual text) plus per-chunk
        # metadata and a relevanceScore. Summaries are not combined with
        # chunk mode, so they are omitted here.
        content_search_spec = {
            "searchResultMode": "CHUNKS",
            "chunkSpec": {
                "numPreviousChunks": num_neighbor_chunks,
                "numNextChunks": num_neighbor_chunks,
            },
        }
    else:
        # DOCUMENTS mode: ask for a generated summary grounded in the retrieved
        # documents. (Standard edition: extractive answers/segments and
        # snippets are not available, so they are intentionally omitted.)
        content_search_spec = {
            "summarySpec": {
                "summaryResultCount": page_size,
                "includeCitations": True,
            }
        }
    body = {
        "query": query,
        "pageSize": page_size,
        "contentSearchSpec": content_search_spec,
    }
    resp = session.post(url, json=body)
    if resp.status_code != 200:
        print(f"Search failed: HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def grounded_generate(
    project: str,
    datastore_location: str,
    datastore_id: str,
    vertex_location: str,
    model: str,
    query: str,
    credentials=None,
) -> dict:
    """Ask Gemini the question grounded in the data store; return raw response JSON.

    Uses the Vertex AI Search retrieval tool so the model answers strictly from
    the indexed handbook. Unlike the ``:search`` summary feature, this works on
    standard-edition data stores.
    """
    datastore = (
        f"projects/{project}/locations/{datastore_location}"
        f"/collections/default_collection/dataStores/{datastore_id}"
    )
    client = Client(
        vertexai=True,
        project=project,
        location=vertex_location,
        credentials=credentials,
    )
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_ai_search=types.VertexAISearch(datastore=datastore)
                    )
                )
            ],
        ),
    )
    return response.model_dump(mode="json", exclude_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to ask the data store.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--datastore-id", default=DEFAULT_DATASTORE_ID)
    parser.add_argument("--page-size", type=int, default=3)
    parser.add_argument("--vertex-location", default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=("search", "grounded", "both"),
        default="both",
        help="search = raw :search API; grounded = Gemini grounded answer; both = run both.",
    )
    parser.add_argument(
        "--result-mode",
        choices=("documents", "chunks"),
        default="documents",
        help=(
            "documents = document-level metadata + summary (default); "
            "chunks = return the actual chunk text (chunk.content) plus metadata."
        ),
    )
    parser.add_argument(
        "--neighbor-chunks",
        type=int,
        default=0,
        help="In --result-mode chunks, number of adjacent chunks to include on each side for context.",
    )
    args = parser.parse_args()

    print(f"\nQuery: {args.query!r}\n", file=sys.stderr)

    session = get_session()

    if args.mode in ("search", "both"):
        response = search(
            session=session,
            project=args.project,
            location=args.location,
            datastore_id=args.datastore_id,
            query=args.query,
            page_size=args.page_size,
            result_mode=args.result_mode,
            num_neighbor_chunks=args.neighbor_chunks,
        )
        print("===== RAW :search RESPONSE =====")
        print(json.dumps(response, indent=2))

    if args.mode in ("grounded", "both"):
        response = grounded_generate(
            project=args.project,
            datastore_location=args.location,
            datastore_id=args.datastore_id,
            vertex_location=args.vertex_location,
            model=args.model,
            query=args.query,
            credentials=session.credentials,
        )
        print("\n===== RAW GROUNDED generate_content RESPONSE =====")
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
