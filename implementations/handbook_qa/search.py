"""Vertex AI Search (`:search`) grounding tool for the handbook QA agent.

Unlike the shared :func:`aieng.agent_evals.tools.vertex_search` tool (which
grounds via Gemini's ``generate_content`` retrieval and surfaces only document
names/URIs), this tool calls the Discovery Engine ``:search`` REST endpoint
directly in ``CHUNKS`` result mode. That returns, in a single call, each
chunk's full text (``chunk.content``) together with the rich per-chunk
``structData`` — ``page``, ``section_heading``, ``document_id`` and the chunk
id — needed both for grounding the answer and for traceability scoring.

``CHUNKS`` mode works on standard-edition data stores (unlike server-side
summaries and extractive segments, which are enterprise-only).

Authentication prefers the attached GCE service account (which has the Discovery
Engine permissions on Coder/GCE workspaces) and falls back to Application
Default Credentials when no metadata server is reachable.
"""

import asyncio
import logging
from typing import Any

import google.auth
import google.auth.transport.requests
from aieng.agent_evals.configs import Configs
from google.adk.tools.function_tool import FunctionTool
from google.auth import compute_engine


logger = logging.getLogger(__name__)

DISCOVERY_ENGINE_BASE = "https://discoveryengine.googleapis.com/v1"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

#: Number of top-ranked chunks to retrieve per search.
DEFAULT_PAGE_SIZE = 5

#: Adjacent chunks to include on either side of each hit for extra context.
_NEIGHBOR_CHUNKS = 0

_session: google.auth.transport.requests.AuthorizedSession | None = None


def _get_session() -> google.auth.transport.requests.AuthorizedSession:
    """Return a cached authorised session, preferring the attached GCE SA.

    The workspace user ADC (from ``gcloud auth application-default login``)
    often lacks the ``serviceusage.services.use`` permission the Discovery
    Engine API needs, so we try the GCE metadata service account first and fall
    back to ADC only if the metadata server is unavailable.
    """
    global _session
    if _session is not None:
        return _session

    request = google.auth.transport.requests.Request()
    try:
        creds = compute_engine.Credentials(scopes=_SCOPES)
        creds.refresh(request)
        logger.debug("handbook_search auth: using attached GCE service account")
    except Exception as exc:  # noqa: BLE001 - fall back to ADC
        logger.debug("handbook_search auth: GCE SA unavailable (%s); using ADC", exc)
        creds, _ = google.auth.default(scopes=_SCOPES)

    _session = google.auth.transport.requests.AuthorizedSession(creds)
    return _session


def _search_sync(datastore: str, query: str, page_size: int) -> list[dict[str, Any]]:
    """Call the ``:search`` endpoint in CHUNKS mode and return the ``results`` list."""
    session = _get_session()
    url = f"{DISCOVERY_ENGINE_BASE}/{datastore}/servingConfigs/default_search:search"
    body = {
        "query": query,
        "pageSize": page_size,
        # CHUNKS mode returns the actual indexed chunk text in ``chunk.content``
        # along with per-chunk metadata and a relevance score, in one request.
        "contentSearchSpec": {
            "searchResultMode": "CHUNKS",
            "chunkSpec": {
                "numPreviousChunks": _NEIGHBOR_CHUNKS,
                "numNextChunks": _NEIGHBOR_CHUNKS,
            },
        },
    }
    resp = session.post(url, json=body)
    resp.raise_for_status()
    return resp.json().get("results", [])


def _doc_id_from_chunk_name(name: str) -> str:
    """Extract the parent document id from a chunk resource name.

    Chunk names look like ``.../documents/{document_id}/chunks/{index}``.
    """
    marker = "/documents/"
    if marker in name:
        return name.split(marker, 1)[1].split("/chunks/", 1)[0]
    return ""


def _parse_chunk(result: dict[str, Any]) -> dict[str, Any]:
    """Map a raw CHUNKS-mode result into a grounded-passage dict."""
    chunk = result.get("chunk", {}) or {}
    struct = (chunk.get("documentMetadata", {}) or {}).get("structData", {}) or {}
    return {
        "content": (chunk.get("content") or "").strip(),
        "page": struct.get("page"),
        "section_heading": struct.get("section_heading", ""),
        "document_name": struct.get("document_name", ""),
        "document_id": struct.get("document_id", ""),
        "chunk_id": _doc_id_from_chunk_name(chunk.get("name", "")),
        "relevance_score": chunk.get("relevanceScore"),
    }


async def handbook_search(query: str) -> dict[str, Any]:
    """Search the LP-Gas O&M Handbook and return grounded passages with citations.

    Use this tool to look up information in the handbook before answering. It
    queries the Vertex AI Search data store, returning the most relevant
    handbook chunks with their full text and the exact source location
    (page number and section heading) so answers can be cited precisely.

    Authentication uses the workspace service account / Application Default
    Credentials — no API key is required.

    Parameters
    ----------
    query : str
        The search query. Be specific and include key terms from the question.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): ``"success"`` or ``"error"``.
        - **results** (list[dict]): Ranked handbook chunks, each with:
            - **content** (str): The full text of the handbook passage.
            - **page** (int | None): 1-based page number in the source PDF.
            - **section_heading** (str): Heading of the section the chunk is in.
            - **document_name** (str): Source document file name.
            - **document_id** (str): Stable document identifier.
            - **chunk_id** (str): Stable identifier of this chunk.
            - **relevance_score** (float | None): Chunk relevance score.
        - **result_count** (int): Number of chunks returned (success case only).
        - **error** (str): Error message (error case only).

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not configured.
    """
    config = Configs()  # type: ignore[call-arg]
    datastore = config.vertex_datastore_id
    if not datastore:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use handbook_search. "
            "Set it in your .env file or as an environment variable."
        )

    try:
        raw_results = await asyncio.to_thread(_search_sync, datastore, query, DEFAULT_PAGE_SIZE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("handbook_search failed: %s", exc)
        return {"status": "error", "error": str(exc), "results": []}

    chunks = [_parse_chunk(result) for result in raw_results]
    return {"status": "success", "results": chunks, "result_count": len(chunks)}


def create_handbook_search_tool(config: Configs | None = None) -> FunctionTool:
    """Create the handbook ``:search`` grounding tool.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, a default ``Configs`` is used.
        Must have ``vertex_datastore_id`` set.

    Returns
    -------
    FunctionTool
        An ADK-compatible tool wrapping :func:`handbook_search`.

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not configured.
    """
    if config is None:
        config = Configs()  # type: ignore[call-arg]

    if not config.vertex_datastore_id:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use create_handbook_search_tool. "
            "Set it in your .env file or as an environment variable."
        )

    return FunctionTool(func=handbook_search)


__all__ = ["create_handbook_search_tool", "handbook_search"]
