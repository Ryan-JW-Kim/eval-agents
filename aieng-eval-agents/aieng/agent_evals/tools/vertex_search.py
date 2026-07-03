"""Vertex AI Search tool for knowledge-grounded QA using a custom data store.

This module provides a search tool that queries a Vertex AI Search data store,
returning grounded summaries with document citations. Unlike the Google Search
tool, content is retrieved by the grounding mechanism — no separate fetch step
is required and no API key is needed (authentication uses ADC).
"""

import logging
import re
from typing import Any

from aieng.agent_evals.configs import Configs
from google.adk.tools.function_tool import FunctionTool
from google.genai import Client, types


logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace to improve regex extraction stability."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_int_string(value: str) -> str:
    """Normalize numeric strings, removing leading zeros."""
    if not value.isdigit():
        return value
    return str(int(value))


def _metadata_page_number_from_uri(uri: str) -> str | None:
    """Read page number from chunk metadata encoded in the document resource name."""
    match = re.search(r"_p(\d+)_", uri)
    if match:
        return _normalize_int_string(match.group(1))
    return None


def _text_inferred_page_number(text: str) -> str | None:
    """Infer page number from chunk text when explicit metadata is missing."""
    if not text:
        return None

    # Direct text marker, e.g. "Page 161".
    text_match = re.search(r"\bPage\s+(\d{1,4})\b", text, flags=re.IGNORECASE)
    if text_match:
        return _normalize_int_string(text_match.group(1))

    normalized = _normalize_whitespace(text)

    # Flattened OCR-like lines:
    # "... Section 8 Maintenance of Fire Protection Equipment 161 Summary ..."
    section_page_match = re.search(
        r"\bSection\s+\d+[A-Za-z]?\b.{0,320}?\b(\d{1,4})\b\s+"
        r"(?:Summary|Purpose|Objectives|General|Maintenance|Checklist|References)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if section_page_match:
        return _normalize_int_string(section_page_match.group(1))

    # Last-resort heuristic: find first plausible page-like number (2-4 digits),
    # excluding common LP-Gas quantities and code references where possible.
    for candidate in re.findall(r"\b(\d{2,4})\b", normalized):
        if candidate in {"1000", "2004", "2002", "2000", "720", "327"}:
            continue
        return _normalize_int_string(candidate)

    return None


def _extract_page_number(uri: str, text: str = "") -> str:
    """Extract page number: metadata first, then text inference, else Unknown."""
    metadata_page = _metadata_page_number_from_uri(uri)
    if metadata_page is not None:
        return metadata_page

    inferred_page = _text_inferred_page_number(text)
    if inferred_page is not None:
        return inferred_page

    return "Unknown"


def _metadata_paragraph_number_from_uri(uri: str) -> str | None:
    """Read paragraph/chunk number from chunk metadata encoded in the resource name."""
    match = re.search(r"_c(\d+)$", uri)
    if match:
        return _normalize_int_string(match.group(1))
    return None


def _text_inferred_paragraph_number(text: str) -> str | None:
    """Infer paragraph number from text when explicit chunk numbering is absent."""
    if not text:
        return None

    text_match = re.search(r"\bParagraph\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if text_match:
        return _normalize_int_string(text_match.group(1))

    # If chunk text exists but no explicit paragraph marker, default to 1.
    if text.strip():
        return "1"

    return None


def _extract_paragraph_number(uri: str, text: str = "") -> str:
    """Extract paragraph number: metadata first, then text inference, else Unknown."""
    metadata_paragraph = _metadata_paragraph_number_from_uri(uri)
    if metadata_paragraph is not None:
        return metadata_paragraph

    inferred_paragraph = _text_inferred_paragraph_number(text)
    if inferred_paragraph is not None:
        return inferred_paragraph

    return "Unknown"


def _text_inferred_section_title(text: str) -> str | None:
    """Infer section title from chunk text when metadata title is generic or absent."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Strict preference for canonical handbook headings.
    for line in lines[:12]:
        if re.match(r"^(Section\s+\d+[A-Za-z]?\s*[-:>].+)$", line):
            return line[:120]
        if re.match(r"^(Appendix\s+[A-Za-z0-9]+\s*[-:>].+)$", line):
            return line[:120]

    # Secondary fallback for plain "Section X" lines.
    for line in lines[:12]:
        if line.lower().startswith("section "):
            return line[:120]

    normalized = _normalize_whitespace(text)

    # Fallback for flattened OCR-like chunks with no line breaks.
    # Capture "Section 8 ..." until a likely page number or known section token.
    flat_match = re.search(
        r"\b(Section\s+\d+[A-Za-z]?(?:\s*[-:>]\s*|\s+)[A-Za-z][A-Za-z0-9&/(),.\-\s]{5,140}?)"
        r"(?:\s+\d{1,4}\b\s+(?:Summary|Purpose|Objectives|General|Maintenance|Checklist|References)\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if flat_match:
        return _normalize_whitespace(flat_match.group(1))[:120]

    # Final text-first fallback: "Section <n> <title words...>" up to page-like number.
    sec_num_match = re.search(r"\bSection\s+(\d+[A-Za-z]?)\b", normalized, flags=re.IGNORECASE)
    if sec_num_match:
        sec_num = sec_num_match.group(1)
        tail = normalized[sec_num_match.end() :]
        words: list[str] = []
        for tok in tail.split():
            if re.fullmatch(r"\d{1,4}", tok):
                break
            cleaned = tok.strip("-:>,.;()")
            if not cleaned:
                continue
            words.append(cleaned)
            if len(words) >= 16:
                break
        if words:
            return f"Section {sec_num} {' '.join(words)}"[:120]

    return None


def _metadata_section_title_from_title(fallback_title: str) -> str | None:
    """Use the chunk metadata title when it carries section-level information."""
    if fallback_title:
        return fallback_title[:120]
    return None


def _extract_section_title(text: str, fallback_title: str) -> str:
    """Extract section title: text inference first, metadata title second, else Unknown."""
    inferred_section = _text_inferred_section_title(text)
    if inferred_section is not None:
        return inferred_section

    metadata_section = _metadata_section_title_from_title(fallback_title)
    if metadata_section is not None:
        return metadata_section

    return "Unknown"


def _short_snippet(text: str, max_words: int = 50) -> str:
    """Return a short snippet of at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _metadata_doc_name_from_title(title: str) -> str | None:
    """Use chunk metadata title to derive a short document name when available."""
    if title:
        return title.split(" > ")[0][:120]
    return None


def _fallback_doc_name_from_uri(uri: str) -> str | None:
    """Fallback document name derived from the document id in the resource name."""
    doc_id = uri.split("/")[-1] if uri else ""
    if doc_id:
        normalized = re.sub(r"_p\d+_c\d+$", "", doc_id)
        normalized = re.sub(r"^doc_", "", normalized)
        return normalized or doc_id
    return None


def _extract_doc_name(title: str, uri: str) -> str:
    """Extract doc name: metadata title first, URI-derived fallback second."""
    metadata_doc_name = _metadata_doc_name_from_title(title)
    if metadata_doc_name is not None:
        return metadata_doc_name

    fallback_doc_name = _fallback_doc_name_from_uri(uri)
    if fallback_doc_name is not None:
        return fallback_doc_name

    return "Unknown"


def _build_citations(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build normalized citation records from grounding chunks."""
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        uri = chunk.get("uri", "")
        text = chunk.get("text", "")
        title = chunk.get("title", "")
        if not text:
            continue

        snippet = _short_snippet(text, max_words=50)
        dedupe_key = (uri, snippet)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        metadata_doc_name = _metadata_doc_name_from_title(title)
        fallback_doc_name = _fallback_doc_name_from_uri(uri)
        if metadata_doc_name is not None:
            doc_name = metadata_doc_name
            doc_name_source = "derived from chunk metadata"
        elif fallback_doc_name is not None:
            doc_name = fallback_doc_name
            doc_name_source = "derived from document URI"
        else:
            doc_name = "Unknown"
            doc_name_source = "unknown source"

        inferred_section = _text_inferred_section_title(text)
        metadata_section = _metadata_section_title_from_title(title)
        if inferred_section is not None:
            section_title = inferred_section
            section_title_source = "inferred from chunk text"
        elif metadata_section is not None:
            section_title = metadata_section
            section_title_source = "derived from chunk metadata"
        else:
            section_title = "Unknown"
            section_title_source = "unknown source"

        metadata_page = _metadata_page_number_from_uri(uri)
        inferred_page = _text_inferred_page_number(text)
        if metadata_page is not None:
            page_number = metadata_page
            page_number_source = "derived from chunk metadata"
        elif inferred_page is not None:
            page_number = inferred_page
            page_number_source = "inferred from chunk text"
        else:
            page_number = "Unknown"
            page_number_source = "unknown source"

        metadata_paragraph = _metadata_paragraph_number_from_uri(uri)
        inferred_paragraph = _text_inferred_paragraph_number(text)
        if metadata_paragraph is not None:
            paragraph_number = metadata_paragraph
            paragraph_number_source = "derived from chunk metadata"
        elif inferred_paragraph is not None:
            paragraph_number = inferred_paragraph
            paragraph_number_source = "inferred from chunk text"
        else:
            paragraph_number = "Unknown"
            paragraph_number_source = "unknown source"

        citations.append(
            {
                "doc_name": doc_name,
                "doc_name_source": doc_name_source,
                "section_title": section_title,
                "section_title_source": section_title_source,
                "page_number": page_number,
                "page_number_source": page_number_source,
                "paragraph_number": paragraph_number,
                "paragraph_number_source": paragraph_number_source,
                "snippet": snippet,
                "uri": uri,
            }
        )

    return citations


def _parse_project_from_datastore_id(datastore_id: str) -> str | None:
    """Parse GCP project ID from a Vertex AI Search data store resource name.

    Parameters
    ----------
    datastore_id : str
        Full resource name, e.g.
        ``projects/my-project/locations/global/collections/default_collection/dataStores/my-store``.

    Returns
    -------
    str or None
        The project ID, or None if the resource name is not in the expected format.
    """
    parts = datastore_id.split("/")
    if len(parts) >= 2 and parts[0] == "projects":
        return parts[1]
    return None


def _extract_datastore_sources(response: Any) -> list[dict[str, str]]:
    """Extract grounding sources from a Vertex AI Search grounded response.

    Vertex AI Search returns ``retrieved_context`` chunks (not ``web`` chunks).
    Each chunk has a ``uri`` (GCS path or document resource name) and an
    optional ``title``.

    Parameters
    ----------
    response : Any
        The Gemini API response object from a Vertex AI Search grounded call.

    Returns
    -------
    list[dict[str, str]]
        List of source dictionaries with ``'title'`` and ``'uri'`` keys.
        Sources with an empty URI are excluded.
    """
    sources: list[dict[str, str]] = []
    if not response.candidates:
        return sources

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if not gm or not hasattr(gm, "grounding_chunks") or not gm.grounding_chunks:
        return sources

    for chunk in gm.grounding_chunks:
        rc = getattr(chunk, "retrieved_context", None)
        if rc:
            # Vertex AI Search returns 'document_name' (full resource path), not 'uri'
            uri = getattr(rc, "document_name", "") or ""
            title = getattr(rc, "title", "") or ""
            if uri:
                sources.append({"title": title, "uri": uri})

    return sources


def _extract_grounding_chunks(response: Any) -> list[dict[str, str]]:
    """Extract the actual text passages used for grounding.

    Each grounding chunk contains the text excerpt from the source document
    that was used to support the generated response.

    Parameters
    ----------
    response : Any
        The Gemini API response object from a Vertex AI Search grounded call.

    Returns
    -------
    list[dict[str, str]]
        List of grounding chunks with ``'text'``, ``'title'``, and ``'uri'`` keys.
    """
    chunks: list[dict[str, str]] = []
    if not response.candidates:
        return chunks

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if not gm or not hasattr(gm, "grounding_chunks") or not gm.grounding_chunks:
        return chunks

    for chunk in gm.grounding_chunks:
        rc = getattr(chunk, "retrieved_context", None)
        if rc:
            # Extract text from retrieved_context
            chunk_text = getattr(rc, "text", "") or ""
            title = getattr(rc, "title", "") or ""
            uri = getattr(rc, "uri", "") or getattr(rc, "document_name", "") or ""
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "title": title,
                    "uri": uri,
                })

    return chunks


async def _vertex_search_async(
    query: str,
    model: str,
    datastore_id: str,
    location: str,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Query a Vertex AI Search data store with grounding enabled.

    Parameters
    ----------
    query : str
        The search query.
    model : str
        The Gemini model to use (accessed via the Vertex AI endpoint).
    datastore_id : str
        Full resource name of the Vertex AI Search data store.
    location : str
        GCP region for the Vertex AI model call (e.g. ``'us-central1'``).
        This is the *compute* region and may differ from the data store's
        ``global`` location.
    temperature : float, default=1.0
        Temperature for generation.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): ``"success"`` or ``"error"``
        - **summary** (str): Grounded text answer drawn from the data store
        - **sources** (list[dict]): Each entry has:
            - **title** (str): Document title
            - **uri** (str): GCS path or Vertex AI document resource name
        - **source_count** (int): Number of sources cited (success case only)
        - **error** (str): Error message (error case only)
    """
    project = _parse_project_from_datastore_id(datastore_id)
    client = Client(vertexai=True, project=project, location=location)
    try:
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(datastore=datastore_id)))
                ],
                temperature=temperature,
            ),
        )

        summary = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    summary += part.text

        sources = _extract_datastore_sources(response)
        chunks = _extract_grounding_chunks(response)
        citations = _build_citations(chunks)
        return {
            "status": "success",
            "summary": summary,
            "sources": sources,
            "source_count": len(sources),
            "grounding_chunks": chunks,
            "citations": citations,
        }

    except Exception as e:
        logger.exception("Vertex AI Search failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "summary": "",
            "sources": [],
        }
    finally:
        client.close()


async def vertex_search(query: str, model: str | None = None) -> dict[str, Any]:
    """Search the custom knowledge base and return grounded results with citations.

    Use this tool to find information from internal documents and knowledge bases.
    Results are grounded directly from retrieved document content — the summary
    is more reliable than web search snippets and no separate fetch step is needed.

    Authentication uses Application Default Credentials (ADC) — no API key is
    required. On GCE/Coder workspaces the attached service account is used
    automatically.

    Parameters
    ----------
    query : str
        The search query. Be specific and include key terms.
    model : str, optional
        The Gemini model to use. Defaults to ``config.default_worker_model``.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): ``"success"`` or ``"error"``
        - **summary** (str): Grounded answer from the knowledge base
        - **sources** (list[dict]): Each with ``'title'`` and ``'uri'``
        - **source_count** (int): Number of sources cited (success case only)
        - **error** (str): Error message (error case only)

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not set in config.

    Examples
    --------
    >>> result = await vertex_search("What is the company leave policy?")
    >>> print(result["summary"])
    >>> for source in result["sources"]:
    ...     print(f"{source['title']}: {source['uri']}")
    """
    config = Configs()  # type: ignore[call-arg]
    if not config.vertex_datastore_id:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use vertex_search. "
            "Set it in your .env file or as an environment variable."
        )
    if model is None:
        model = config.default_worker_model

    return await _vertex_search_async(
        query,
        model=model,
        datastore_id=config.vertex_datastore_id,
        location=config.google_cloud_location,
        temperature=config.default_temperature,
    )


def create_vertex_search_tool(config: Configs | None = None) -> FunctionTool:
    """Create a search tool backed by a custom Vertex AI Search data store.

    Authentication uses Application Default Credentials (ADC) — no API key is
    needed. On GCE/Coder workspaces the attached service account handles auth
    automatically.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
        Must have ``vertex_datastore_id`` set.

    Returns
    -------
    FunctionTool
        An ADK-compatible tool that returns grounded summaries with citations.

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not set in config.

    Examples
    --------
    >>> from aieng.agent_evals.tools import create_vertex_search_tool
    >>> tool = create_vertex_search_tool()
    >>> agent = Agent(tools=[tool])
    """
    if config is None:
        config = Configs()  # type: ignore[call-arg]

    if not config.vertex_datastore_id:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use create_vertex_search_tool. "
            "Set it in your .env file or as an environment variable."
        )

    return FunctionTool(func=vertex_search)
