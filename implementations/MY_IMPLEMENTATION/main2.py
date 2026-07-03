import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv
from google.auth import compute_engine
from requests.exceptions import RequestException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aieng.agent_evals.configs import Configs
from implementations.MY_IMPLEMENTATION.langfuse_tracing import (
    compact_text,
    get_current_observation_id,
    get_current_trace_id,
    get_langfuse_settings,
    langfuse_generation,
    langfuse_span,
    safe_flush_langfuse,
    safe_update_observation,
    safe_update_trace,
    truncate_value,
)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
DISCOVERY_ENGINE_BASE = "https://discoveryengine.googleapis.com/v1"
DEFAULT_PAGE_SIZE = 30
DEFAULT_MAX_PAGES = 4
DEFAULT_MAX_CHUNKS = 80
DEFAULT_MAX_EVIDENCE_SENTENCES = 4

CONSOLE = Console(width=100)


@dataclass
class SearchChunk:
    text: str
    title: str
    uri: str
    metadata: dict[str, Any]
    document_name: str
    document_id: str
    relevance_score: float = 0.0


@dataclass
class EvidenceSentence:
    sentence: str
    score: int
    overlap: int
    anchor_hits: int
    has_number: bool
    has_units: bool
    is_action: bool
    chunk: SearchChunk
    sentence_index: int


def _debug_enabled() -> bool:
    return os.getenv("VERTEX_CITATION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _debug_enabled():
        CONSOLE.print(f"[dim][citation-debug][/dim] {message}")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        CONSOLE.print(f"[yellow]Invalid {name}={raw!r}; using {default}[/yellow]")
        return default


def _trace_identity() -> tuple[str | None, str | None]:
    user_id = os.getenv("LANGFUSE_USER_ID", "").strip() or None
    session_id = os.getenv("LANGFUSE_SESSION_ID", "").strip() or None
    return user_id, session_id


def _trace_metadata(query: str, config: Configs) -> dict[str, Any]:
    project_id = None
    datastore_short_id = None
    try:
        project_id, datastore_short_id = _parse_datastore_resource(config.vertex_datastore_id or "")
    except Exception:  # noqa: BLE001
        project_id = None
        datastore_short_id = None

    settings = get_langfuse_settings()
    return {
        "app_env": settings.app_env,
        "app_version": settings.app_version,
        "datastore_id": config.vertex_datastore_id,
        "datastore_project_id": project_id,
        "datastore_short_id": datastore_short_id,
        "google_cloud_location": str(config.google_cloud_location),
        "query": compact_text(query, 600),
        "query_preview": compact_text(query, 240),
        "query_length": len(query.strip()),
        "capture_full_prompt": settings.capture_full_prompt,
    }


def _citation_trace_summary(citations: list[dict[str, Any]]) -> dict[str, Any]:
    section_headings = [str(c.get("section_title", "")) for c in citations]
    pages = [c.get("page_number") for c in citations]
    chunk_ids = [str(c.get("paragraph_number", "")) for c in citations]
    unknown_section = any(not str(c.get("section_title", "")).strip() or str(c.get("section_title", "")).strip().lower() == "unknown" for c in citations)
    has_source_text = [bool(str(c.get("source_text", "")).strip()) for c in citations]
    return {
        "citation_count": len(citations),
        "citation_ids": [str(c.get("uri", "")) for c in citations],
        "section_headings": section_headings,
        "pages": pages,
        "chunk_ids": chunk_ids,
        "unknown_section_heading": unknown_section,
        "unknown_section_heading_count": sum(
            1
            for c in citations
            if not str(c.get("section_title", "")).strip()
            or str(c.get("section_title", "")).strip().lower() == "unknown"
        ),
        "has_source_text": has_source_text,
    }


def _safe_tag_value(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_\-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for block in re.split(r"\n+|\u2022", text):
        block = block.strip()
        if not block:
            continue
        pieces.extend(re.split(r"(?<=[.!?])\s+", block))

    cleaned: list[str] = []
    for piece in pieces:
        sentence = re.sub(r"\s+", " ", piece).strip()
        if re.match(r"^\d+(?:\.\d+)+\s*-", sentence):
            continue
        if re.match(r"^Section\s+\d+\s*-\s*.+\s+\d+$", sentence, flags=re.IGNORECASE):
            continue
        if len(sentence) >= 12:
            cleaned.append(sentence)
    return cleaned


def _short_snippet(text: str, max_words: int = 50) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _is_likely_out_of_scope(query: str) -> bool:
    q = query.lower()
    direct_terms = {
        "vacation",
        "salary",
        "payroll",
        "pto",
        "leave benefits",
        "insurance premium",
        "thermostat",
        "btu",
        "furnace",
        "distillation",
        "refinery",
        "fractionation",
        "crude",
        "cdl",
        "commercial driver's license",
        "driver license",
    }
    if any(term in q for term in direct_terms):
        return True

    if re.search(r"\b(price|pricing|retail market price)\b", q) and re.search(r"\b(gallon|propane)\b", q):
        return True

    return False


def _is_likely_unsafe_request(query: str) -> bool:
    q = query.lower()
    terms = {
        "disable",
        "bypass",
        "defeat",
        "ignore",
        "keep loading anyway",
        "remove the valve",
        "plug the relief",
        "without requalifying",
        "past its requalification date",
        "force the cap off",
        "fill past",
        "skip notifying",
        "overfill",
        "keep filling",
        "vent a tank",
        "vent to the atmosphere",
    }
    return any(term in q for term in terms)


def _is_likely_procedural_query(query: str) -> bool:
    q = query.lower()
    direct_terms = {
        "procedure",
        "steps",
        "what must",
        "during",
        "before",
        "after",
        "if a leak",
        "if a fire",
        "how do",
        "how is",
        "how should",
        "how can",
        "how does",
    }
    if any(term in q for term in direct_terms):
        return True

    if re.search(r"\bhow\b", q) and not _is_likely_numeric_query(query):
        return True

    return False


def _is_likely_synthesis_query(query: str) -> bool:
    q = query.lower()
    terms = {
        "combine",
        "combining",
        "together",
        "across",
        "integrate",
        "compare",
        "versus",
        "and also",
    }
    return any(term in q for term in terms)


def _is_likely_table_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in {"table", "chart", "row", "column", "label", "placard", "indicate"})


def _is_likely_numeric_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(how many|how far|how close|what distance|within what|minimum|maximum|percent|percentage|years?)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _is_multi_part_query(query: str) -> bool:
    q = query.lower()
    if any(term in q for term in {"what two", "what three", "which editions", "where must", "what must be in place"}):
        return True
    if re.search(r"\b(and|both|along with|as well as)\b", q):
        return True
    if q.count(",") >= 1:
        return True
    return False


def _refusal_summary(query: str) -> str:
    q = query.lower()
    if _is_likely_unsafe_request(query):
        if "requalification" in q:
            return (
                "I can't help with refilling a cylinder past its requalification date. "
                "It must be requalified before refilling; stop the operation and notify your supervisor."
            )
        if "white mist" in q or "maximum liquid level" in q:
            return (
                "I can't help with overfilling. "
                "At maximum fill you must close the hose end valve and shut off transfer."
            )
        if "thermal" in q or "emergency shutoff" in q:
            return (
                "I can't help with disabling required thermal or emergency shutoff safety devices. "
                "Do not defeat safety controls; notify your supervisor for repair."
            )
        if "cap" in q:
            return (
                "I can't help with forcing that cap off. "
                "If discharge continues, do not remove the cap and get supervisor assistance."
            )
        if "vent" in q or "atmosphere" in q or "fire department" in q:
            return (
                "I can't help with venting propane to the atmosphere. "
                "Use approved evacuation procedures, flare only when permitted, and follow fire department notification rules."
            )
        return (
            "I can't help with unsafe instructions. "
            "These are required safety controls; stop the operation and notify your supervisor."
        )

    return (
        "That topic is outside this handbook's scope. "
        "This handbook covers LP-Gas bulk plant operations, maintenance, safety, and emergency procedures."
    )


def _infer_safety_level(query: str, answer_text: str) -> str:
    q = query.lower()
    a = answer_text.lower()
    text = f"{q} {a}"

    if _is_likely_unsafe_request(query):
        return "critical"
    if _is_likely_out_of_scope(query):
        return "negligible"

    critical_terms = {"explosion", "bleve", "fatal", "major release", "venting propane", "atmosphere"}
    high_terms = {
        "fire",
        "leak",
        "emergency shutoff",
        "relief valve",
        "thermal link",
        "liquid transfer",
        "evacuat",
        "odorization",
        "overfill",
        "fire extinguisher",
        "smoking",
    }
    moderate_terms = {"inspection", "maintenance", "checklist", "clearance", "parking brake", "chock", "service valve"}
    low_terms = {"record", "documentation", "frequency", "edition", "chapter", "requalification", "years"}

    if any(term in text for term in critical_terms):
        return "critical"
    if sum(1 for term in high_terms if term in text) >= 1:
        return "high"
    if sum(1 for term in moderate_terms if term in text) >= 1:
        return "moderate"
    if sum(1 for term in low_terms if term in text) >= 1:
        return "low"
    return "negligible"


def _query_variants(query: str) -> list[str]:
    variants = [query.strip()]

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "and",
        "or",
        "if",
        "how",
        "what",
        "when",
        "where",
        "which",
        "that",
        "this",
        "these",
        "those",
        "must",
        "can",
        "could",
        "would",
        "should",
        "do",
        "does",
        "did",
        "it",
        "its",
        "as",
        "with",
        "by",
        "than",
        "then",
        "into",
        "about",
        "awaiting",
        "use",
        "resale",
        "exchange",
        "location",
        "storage",
    }

    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    keywords = [t for t in tokens if t not in stopwords]
    if keywords:
        keyword_query = " ".join(keywords[:16])
        if keyword_query and keyword_query not in variants:
            variants.append(keyword_query)

    numeric_tokens = re.findall(r"\d+(?:\.\d+)?", query)
    if numeric_tokens and keywords:
        numeric_query = " ".join((keywords[:12] + numeric_tokens)[:18])
        if numeric_query and numeric_query not in variants:
            variants.append(numeric_query)

    return variants


def _extract_constraint_phrases(sentence: str) -> list[str]:
    constraints: list[str] = []

    numeric_patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:feet?|ft)\b",
        r"\b\d+(?:\.\d+)?\s*(?:lb|pounds?)\b",
        r"\b\d+(?:\.\d+)?\s*(?:years?)\b",
        r"\b\d+(?:\.\d+)?\s*%\b",
    ]
    for pattern in numeric_patterns:
        for match in re.findall(pattern, sentence, flags=re.IGNORECASE):
            constraints.append(match.strip())

    action_clauses = [
        r"\bclose\s+[^.,;]{3,60}",
        r"\bshut\s+down\s+[^.,;]{3,60}",
        r"\bdo\s+not\s+[^.,;]{3,60}",
        r"\bmust\s+not\s+[^.,;]{3,60}",
        r"\bnotify\s+[^.,;]{3,60}",
        r"\bevacuat\w*\s+[^.,;]{3,60}",
    ]
    for pattern in action_clauses:
        for match in re.findall(pattern, sentence, flags=re.IGNORECASE):
            constraints.append(re.sub(r"\s+", " ", match.strip()))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in constraints:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_datastore_resource(datastore_resource: str) -> tuple[str, str]:
    parts = datastore_resource.split("/")
    if len(parts) < 8 or parts[0] != "projects" or "dataStores" not in parts:
        raise ValueError(
            "Invalid VERTEX_AI_DATASTORE_ID format. Expected: "
            "projects/{project}/locations/global/collections/default_collection/dataStores/{id}"
        )

    ds_index = parts.index("dataStores")
    if ds_index + 1 >= len(parts):
        raise ValueError("Invalid VERTEX_AI_DATASTORE_ID: missing datastore id")

    return parts[1], parts[ds_index + 1]


def _extract_title_from_document(struct_data: dict[str, Any], derived: dict[str, Any]) -> str:
    for key in ("title", "section_heading", "document_title", "name"):
        value = struct_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = derived.get("title")
    if isinstance(value, str) and value.strip():
        return value.strip()

    return ""


def _extract_metadata_text(struct_data: dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content", "body"):
        value = struct_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _looks_like_doc_identifier(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    return bool(
        re.match(r"^doc_[a-z0-9_]+_p\d+_c\d+$", normalized)
        or re.match(r"^[a-f0-9]{16,}$", normalized)
    )


def _collect_nested_candidates(metadata: dict[str, Any], wanted_keys: set[str], max_depth: int = 4) -> list[str]:
    candidates: list[str] = []

    def walk(node: Any, depth: int, key_hint: str = "") -> None:
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, depth + 1, str(key))
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth + 1, key_hint)
            return
        if isinstance(node, str) and key_hint.lower() in wanted_keys:
            stripped = node.strip()
            if stripped:
                candidates.append(stripped)

    walk(metadata, 0)
    return candidates


def _normalize_search_results(raw_results: list[dict[str, Any]]) -> list[SearchChunk]:
    chunks: list[SearchChunk] = []

    for result in raw_results:
        if not isinstance(result, dict):
            continue

        chunk_data = result.get("chunk")
        if isinstance(chunk_data, dict):
            text = str(chunk_data.get("content") or "").strip()
            chunk_name = str(chunk_data.get("name") or "")
            document_metadata = chunk_data.get("documentMetadata", {})
            if not isinstance(document_metadata, dict):
                document_metadata = {}
            struct_data = document_metadata.get("structData", {})
            if not isinstance(struct_data, dict):
                struct_data = {}

            doc_match = re.search(r"/documents/([^/]+)/chunks/", chunk_name)
            doc_id = doc_match.group(1) if doc_match else "unknown"

            doc_page_match = re.search(r"_p(\d+)_", doc_id)
            if doc_page_match:
                page_num = doc_page_match.group(1)
            else:
                chunk_num_match = re.search(r"/chunks/(\d+)$", chunk_name)
                page_num = chunk_num_match.group(1) if chunk_num_match else "Unknown"

            title = _extract_title_from_document(struct_data, document_metadata)
            if not text:
                text = _extract_metadata_text(struct_data)

            metadata: dict[str, Any] = dict(struct_data)
            for key in (
                "title",
                "section",
                "section_title",
                "sectionTitle",
                "heading",
                "header",
                "chapter",
                "chapter_title",
            ):
                if key in document_metadata and document_metadata.get(key) is not None:
                    metadata[key] = document_metadata.get(key)
            metadata["documentMetadata"] = document_metadata
            metadata["page"] = page_num
            metadata["document_id"] = doc_id
            metadata["document_name"] = doc_id

            _debug(
                "raw chunk metadata "
                f"uri={chunk_name!r} "
                f"chunk_keys={sorted(chunk_data.keys())} "
                f"document_metadata_keys={sorted(document_metadata.keys())} "
                f"struct_data_keys={sorted(struct_data.keys())}"
            )

            chunks.append(
                SearchChunk(
                    text=text,
                    title=title,
                    uri=chunk_name,
                    metadata=metadata,
                    document_name=re.sub(r"/chunks/[^/]+$", "", chunk_name),
                    document_id=doc_id,
                    relevance_score=float(chunk_data.get("relevanceScore") or 0.0),
                )
            )
            continue

        document = result.get("document")
        if not isinstance(document, dict):
            continue

        struct_data = document.get("structData")
        if not isinstance(struct_data, dict):
            struct_data = {}

        derived = document.get("derivedStructData")
        if not isinstance(derived, dict):
            derived = {}

        document_name = str(document.get("name") or "")
        chunks.append(
            SearchChunk(
                text=_extract_metadata_text(struct_data),
                title=_extract_title_from_document(struct_data, derived),
                uri=str(derived.get("link") or document_name),
                metadata=struct_data,
                document_name=document_name,
                document_id=str(document.get("id") or ""),
                relevance_score=0.0,
            )
        )

    deduped: list[SearchChunk] = []
    seen_uri: set[str] = set()
    for chunk in chunks:
        if chunk.uri and chunk.uri in seen_uri:
            continue
        if chunk.uri:
            seen_uri.add(chunk.uri)
        deduped.append(chunk)
    return deduped


def _query_term_bigrams(query: str) -> set[str]:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _query_anchor_tokens(query: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    generic = {
        "what",
        "which",
        "when",
        "where",
        "how",
        "many",
        "close",
        "must",
        "should",
        "could",
        "would",
        "bulk",
        "plant",
        "lp",
        "gas",
        "propane",
        "operations",
        "safety",
        "requirements",
        "section",
        "handbook",
        "during",
        "before",
        "after",
        "within",
        "minimum",
        "maximum",
    }
    anchors = {t for t in tokens if len(t) > 3 and t not in generic}
    return anchors


def _score_chunk(query: str, chunk: SearchChunk, query_tokens: set[str], query_bigrams: set[str]) -> int:
    text = chunk.text
    text_tokens = _tokenize(text)
    overlap = len(query_tokens & text_tokens)
    bigram_hits = sum(1 for bg in query_bigrams if bg in text.lower())

    numeric_in_query = bool(re.search(r"\d", query))
    numeric_in_chunk = bool(re.search(r"\d", text))
    action_terms = {"close", "stop", "disconnect", "evacuat", "notify", "inspect", "shutdown", "manual", "valve"}
    action_hits = len(action_terms & text_tokens)

    score = (6 * overlap) + (5 * bigram_hits) + (3 * action_hits)
    if numeric_in_query and numeric_in_chunk:
        score += 6
    if _is_likely_table_query(query) and re.search(r"\b(table|row|column|figure|\d+\.\d+|\d+,\d+)\b", text.lower()):
        score += 6

    score += int(chunk.relevance_score * 5)
    return score


def _rank_chunks(query: str, chunks: list[SearchChunk]) -> list[SearchChunk]:
    query_tokens = _tokenize(query)
    query_bigrams = _query_term_bigrams(query)

    ranked: list[tuple[int, int, SearchChunk]] = []
    for chunk in chunks:
        score = _score_chunk(query, chunk, query_tokens, query_bigrams)
        chunk.metadata["rank_score"] = score
        ranked.append((score, len(chunk.text), chunk))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def _score_sentence(
    query: str,
    sentence: str,
    query_tokens: set[str],
    query_numbers: set[str],
    query_anchors: set[str],
) -> tuple[int, int, int, bool, bool, bool]:
    sentence_tokens = _tokenize(sentence)
    overlap = len(query_tokens & sentence_tokens)

    sentence_numbers = set(re.findall(r"\d+(?:\.\d+)?", sentence))
    number_overlap = len(query_numbers & sentence_numbers)
    has_number = bool(sentence_numbers)
    has_units = bool(re.search(r"\b(feet?|ft|lb|pounds?|psi|inch|inches|year|years|%|gallon|psig|inch)\b", sentence, re.IGNORECASE))

    action_terms = {
        "close",
        "stop",
        "disconnect",
        "evacuate",
        "notify",
        "inspect",
        "record",
        "chock",
        "secure",
        "shutdown",
        "isolate",
        "must",
        "shall",
    }
    is_action = len(action_terms & sentence_tokens) > 0
    anchor_hits = len(query_anchors & sentence_tokens)

    score = (6 * overlap) + (10 * number_overlap) + (4 * int(has_units)) + (4 * int(is_action))
    score += 7 * anchor_hits

    if _is_likely_numeric_query(query) and has_number:
        score += 7
    if _is_likely_table_query(query) and len(sentence_numbers) >= 2:
        score += 5
    if _is_likely_procedural_query(query) and is_action:
        score += 4

    if len(sentence) < 20:
        score -= 3
    if query_anchors and anchor_hits == 0:
        score -= 8

    return score, overlap, anchor_hits, has_number, has_units, is_action


def _collect_evidence(query: str, ranked_chunks: list[SearchChunk]) -> list[EvidenceSentence]:
    query_tokens = _tokenize(query)
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
    query_anchors = _query_anchor_tokens(query)

    max_chunks = _get_int_env("VERTEX_MAX_RANKED_CHUNKS", DEFAULT_MAX_CHUNKS)
    candidates: list[EvidenceSentence] = []

    for chunk in ranked_chunks[:max_chunks]:
        if not chunk.text.strip():
            continue

        for i, sentence in enumerate(_split_sentences(chunk.text)):
            score, overlap, anchor_hits, has_number, has_units, is_action = _score_sentence(
                query,
                sentence,
                query_tokens,
                query_numbers,
                query_anchors,
            )
            if score <= 0:
                continue
            candidates.append(
                EvidenceSentence(
                    sentence=sentence,
                    score=score,
                    overlap=overlap,
                    anchor_hits=anchor_hits,
                    has_number=has_number,
                    has_units=has_units,
                    is_action=is_action,
                    chunk=chunk,
                    sentence_index=i,
                )
            )

    candidates.sort(key=lambda item: (item.score, len(item.sentence)), reverse=True)

    deduped: list[EvidenceSentence] = []
    seen_norm: set[str] = set()
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate.sentence.lower())
        if normalized in seen_norm:
            continue
        seen_norm.add(normalized)
        deduped.append(candidate)
        if len(deduped) >= 40:
            break

    return deduped


def _select_evidence(query: str, evidence: list[EvidenceSentence]) -> list[EvidenceSentence]:
    if not evidence:
        return []

    is_numeric = _is_likely_numeric_query(query)
    is_procedural = _is_likely_procedural_query(query)
    is_synthesis = _is_likely_synthesis_query(query)
    is_table = _is_likely_table_query(query)
    is_multi_part = _is_multi_part_query(query)

    if is_numeric and not is_procedural and not is_synthesis:
        max_numeric = 3 if is_multi_part else 2
        selected = [e for e in evidence if e.has_number and e.has_units and e.anchor_hits >= 1 and e.overlap >= 2][:max_numeric]
        if not selected:
            selected = [e for e in evidence if e.has_number and e.anchor_hits >= 1 and e.overlap >= 1][:max_numeric]
        if not selected:
            selected = evidence[:1]
        return selected

    if is_table:
        selected = [e for e in evidence if e.has_number and e.overlap >= 1][:2]
        if len(selected) < 2:
            selected.extend([e for e in evidence if e not in selected][: 2 - len(selected)])
        return selected[:2]

    if is_procedural or is_synthesis:
        selected = [e for e in evidence if e.is_action and e.overlap >= 2][:3]
        if len(selected) < 2:
            selected.extend([e for e in evidence if e not in selected and e.overlap >= 1][: 2 - len(selected)])
        return selected[:3]

    max_factual = 3 if is_multi_part else 2
    selected = [e for e in evidence if e.anchor_hits >= 1 and e.overlap >= 2][:max_factual]
    if selected:
        return selected
    return evidence[:1]


def _extract_section_title(text: str, title: str, metadata: dict[str, Any]) -> tuple[str, str]:
    direct_keys = (
        "section",
        "section_title",
        "sectionTitle",
        "section_heading",
        "heading",
        "header",
        "section_name",
        "sectionName",
        "section_header",
        "sectionHeader",
        "chapter",
        "chapter_title",
        "chapterTitle",
        "title",
        "document_title",
        "documentTitle",
        "name",
    )

    for key in direct_keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            candidate = str(value).strip()[:140]
            if not _looks_like_doc_identifier(candidate):
                return candidate, "derived from chunk metadata"

    nested_key_set = {k.lower() for k in direct_keys}
    for candidate in _collect_nested_candidates(metadata, nested_key_set):
        if not _looks_like_doc_identifier(candidate):
            return candidate[:140], "derived from nested chunk metadata"

    if title and not _looks_like_doc_identifier(title):
        return title[:140], "derived from chunk metadata"

    for line in [line.strip() for line in text.splitlines() if line.strip()][:12]:
        if re.match(r"^(Section\s+\d+[A-Za-z]?\s*[-:>].+)$", line):
            return line[:140], "inferred from chunk text"
        if re.match(r"^(Appendix\s+[A-Za-z0-9]+\s*[-:>].+)$", line):
            return line[:140], "inferred from chunk text"

    section_match = re.search(r"\b(Section\s+\d+[A-Za-z]?\s*[-:>]??\s*[^\n.]{3,100})", text)
    if section_match:
        return section_match.group(1).strip()[:140], "inferred from chunk text"

    appendix_match = re.search(r"\b(Appendix\s+[A-Za-z0-9]+\s*[-:>]??\s*[^\n.]{3,100})", text)
    if appendix_match:
        return appendix_match.group(1).strip()[:140], "inferred from chunk text"

    return "Unknown", "unknown source"


def _extract_doc_name(title: str, uri: str, metadata: dict[str, Any]) -> tuple[str, str]:
    for key in ("doc_name", "document_name", "document", "source_document", "file_name", "filename"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()[:140], "derived from chunk metadata"

    if title:
        return title.split(" > ")[0][:140], "derived from chunk metadata"

    doc_id = uri.split("/")[-1] if uri else ""
    if doc_id:
        normalized = re.sub(r"_p\d+_c\d+$", "", doc_id)
        normalized = re.sub(r"^doc_", "", normalized)
        return normalized or doc_id, "derived from document URI"

    return "Unknown", "unknown source"


def _normalize_int_string(value: str) -> str:
    return str(int(value)) if value.isdigit() else value


def _extract_page_number(uri: str, text: str, metadata: dict[str, Any]) -> tuple[str, str]:
    for key in ("page", "page_number", "pageNumber", "pageIdentifier"):
        value = metadata.get(key)
        if value is not None and str(value).strip() and str(value).strip() != "Unknown":
            return _normalize_int_string(str(value).strip()), "derived from chunk metadata"

    match = re.search(r"_p(\d+)_", uri)
    if match:
        return _normalize_int_string(match.group(1)), "derived from chunk metadata"

    match = re.search(r"\bPage\s+(\d{1,4})\b", text, flags=re.IGNORECASE)
    if match:
        return _normalize_int_string(match.group(1)), "inferred from chunk text"

    return "Unknown", "unknown source"


def _extract_paragraph_number(uri: str, text: str, metadata: dict[str, Any]) -> tuple[str, str]:
    for key in ("paragraph", "paragraph_number", "paragraphNumber", "chunk", "chunk_number"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _normalize_int_string(str(value).strip()), "derived from chunk metadata"

    match = re.search(r"_c(\d+)$", uri)
    if match:
        return _normalize_int_string(match.group(1)), "derived from chunk metadata"

    match = re.search(r"\bParagraph\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if match:
        return _normalize_int_string(match.group(1)), "inferred from chunk text"

    if text.strip():
        return "1", "inferred from chunk text"

    return "Unknown", "unknown source"


def _build_citations(selected_evidence: list[EvidenceSentence]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for evidence in selected_evidence:
        chunk = evidence.chunk
        uri = chunk.uri
        text = chunk.text
        title = chunk.title
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}

        snippet = _short_snippet(evidence.sentence, max_words=50)
        dedupe_key = (uri, snippet)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        _debug(
            "citation input "
            f"uri={uri!r} "
            f"title={title!r} "
            f"metadata_keys={sorted(metadata.keys())} "
            f"support_score={evidence.score}"
        )

        doc_name, doc_name_source = _extract_doc_name(title, uri, metadata)
        section_title, section_title_source = _extract_section_title(text, title, metadata)
        page_number, page_number_source = _extract_page_number(uri, text, metadata)
        paragraph_number, paragraph_number_source = _extract_paragraph_number(uri, text, metadata)

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
                "source_text": evidence.sentence,
                "support_score": evidence.score,
                "uri": uri,
            }
        )

    return citations


def _annotate_summary_with_citations(summary: str, citations: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if not summary or not citations:
        return summary, []

    sentences = _split_sentences(summary)
    citation_tokens = [_tokenize(str(c.get("snippet", ""))) for c in citations]

    index_by_citation: dict[int, int] = {}
    ordered_used: list[dict[str, Any]] = []
    annotated_sentences: list[str] = []

    for sentence_idx, sentence in enumerate(sentences):
        sentence_tokens = _tokenize(sentence)
        best_idx: int | None = None
        best_score = 0

        for i, c_tokens in enumerate(citation_tokens):
            score = len(sentence_tokens & c_tokens)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None or best_score == 0:
            best_idx = sentence_idx % len(citations)

        if best_idx not in index_by_citation:
            index_by_citation[best_idx] = len(index_by_citation) + 1
            ordered_used.append(citations[best_idx])

        marker = index_by_citation[best_idx]
        annotated_sentences.append(f"{sentence} [{marker}]")

    return " ".join(annotated_sentences), ordered_used


def _assemble_answer(query: str, selected_evidence: list[EvidenceSentence]) -> dict[str, Any]:
    if not selected_evidence:
        return {
            "answer_text": "",
            "safety_level": _infer_safety_level(query, ""),
            "constraints": [],
            "insufficient_evidence": True,
        }

    is_procedural = _is_likely_procedural_query(query)
    is_synthesis = _is_likely_synthesis_query(query)

    if is_procedural or is_synthesis:
        # Keep deterministic ordering for procedural answers.
        ordered = sorted(selected_evidence, key=lambda e: (e.chunk.metadata.get("page", "9999"), e.sentence_index))
    else:
        ordered = selected_evidence

    answer_text = " ".join(e.sentence for e in ordered)
    answer_text = re.sub(r"\s+", " ", answer_text).strip()

    constraints: list[str] = []
    for evidence in ordered:
        constraints.extend(_extract_constraint_phrases(evidence.sentence))
    constraints = list(dict.fromkeys(constraints))[:5]

    return {
        "answer_text": answer_text,
        "safety_level": _infer_safety_level(query, answer_text),
        "constraints": constraints,
        "insufficient_evidence": False,
    }


def get_credentials() -> google.auth.credentials.Credentials:
    request = google.auth.transport.requests.Request()

    try:
        creds = compute_engine.Credentials(scopes=SCOPES)
        creds.refresh(request)
        CONSOLE.print("[dim]Auth: using attached GCE service account[/dim]")
        return creds
    except Exception:
        CONSOLE.print("[dim]Auth: attached GCE service account unavailable; using ADC[/dim]")

    creds, _ = google.auth.default(scopes=SCOPES)
    return creds


def search_discovery_engine(
    query: str,
    config: Configs,
    credentials: google.auth.credentials.Credentials,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not config.vertex_datastore_id:
        raise ValueError("VERTEX_AI_DATASTORE_ID must be set.")

    settings = get_langfuse_settings()
    trace_metadata = _trace_metadata(query, config)
    trace_context = trace_context or {}
    evaluation_item_id = str(trace_context.get("evaluation_item_id", "")).strip()
    evaluation_category = str(trace_context.get("evaluation_category", "")).strip()
    evaluation_expected_behavior = str(trace_context.get("evaluation_expected_behavior", "")).strip()
    evaluation_run_trace_id = str(trace_context.get("evaluation_run_trace_id", "")).strip()
    evaluation_run_observation_id = str(trace_context.get("evaluation_run_observation_id", "")).strip()

    if evaluation_item_id:
        trace_metadata["evaluation_item_id"] = evaluation_item_id
    if evaluation_category:
        trace_metadata["evaluation_category"] = evaluation_category
    if evaluation_expected_behavior:
        trace_metadata["evaluation_expected_behavior"] = evaluation_expected_behavior
    if evaluation_run_trace_id:
        trace_metadata["evaluation_run_trace_id"] = evaluation_run_trace_id
    if evaluation_run_observation_id:
        trace_metadata["evaluation_run_observation_id"] = evaluation_run_observation_id

    user_id, session_id = _trace_identity()
    root_trace_id: str | None = None
    root_observation_id: str | None = None

    with langfuse_span(
        name="pdf_qa_pipeline",
        input=truncate_value(
            {
                "query": query,
                "query_preview": compact_text(query, 240),
                "datastore_id": config.vertex_datastore_id,
            }
        ),
        metadata=trace_metadata,
        version=settings.app_version,
    ) as pipeline_span:
        decision_type = "answer"
        initial_tags = [
            "pdf_qa",
            "vertex_search",
            f"app_env:{_safe_tag_value(str(settings.app_env or 'unknown'))}",
            "evaluation" if evaluation_item_id else "interactive",
            f"category:{_safe_tag_value(evaluation_category)}" if evaluation_category else "",
            f"expected_behavior:{_safe_tag_value(evaluation_expected_behavior)}" if evaluation_expected_behavior else "",
        ]
        safe_update_trace(
            name="pdf_qa_pipeline",
            user_id=user_id,
            session_id=session_id,
            version=settings.app_version,
            metadata=trace_metadata,
            tags=[tag for tag in initial_tags if tag],
        )
        root_trace_id = get_current_trace_id()
        root_observation_id = get_current_observation_id()

        project_id, datastore_short_id = _parse_datastore_resource(config.vertex_datastore_id)
        serving_configs = [
            f"projects/{project_id}/locations/global/collections/default_collection/dataStores/{datastore_short_id}/servingConfigs/default_search",
            f"projects/{project_id}/locations/global/collections/default_collection/dataStores/{datastore_short_id}/servingConfigs/default_config",
        ]

        page_size = _get_int_env("VERTEX_SEARCH_PAGE_SIZE", DEFAULT_PAGE_SIZE)
        max_pages = _get_int_env("VERTEX_SEARCH_MAX_PAGES", DEFAULT_MAX_PAGES)
        max_evidence = _get_int_env("VERTEX_MAX_EVIDENCE_SENTENCES", DEFAULT_MAX_EVIDENCE_SENTENCES)

        session = google.auth.transport.requests.AuthorizedSession(credentials)
        last_error: str | None = None
        final_status = "error"

        with langfuse_span(
            name="classify_request",
            input={"query": compact_text(query, 240)},
            metadata={"query_preview": compact_text(query, 240)},
            version=settings.app_version,
        ) as classify_span:
            unsafe_request = _is_likely_unsafe_request(query)
            out_of_scope = _is_likely_out_of_scope(query)
            procedural_query = _is_likely_procedural_query(query)
            synthesis_query = _is_likely_synthesis_query(query)
            table_query = _is_likely_table_query(query)
            numeric_query = _is_likely_numeric_query(query)
            classification = {
                "unsafe_request": unsafe_request,
                "out_of_scope": out_of_scope,
                "query_type": {
                    "procedural": procedural_query,
                    "synthesis": synthesis_query,
                    "table": table_query,
                    "numeric": numeric_query,
                },
            }
            safe_update_observation(classify_span, output=classification, metadata=classification)

        if unsafe_request or out_of_scope:
            refusal = _refusal_summary(query)
            final_status = "refused" if unsafe_request else "no_evidence"
            decision_type = "unsafe" if unsafe_request else "out_of_scope"
            refusal_safety = _infer_safety_level(query, refusal)
            safe_update_trace(
                output={"status": final_status, "summary": refusal},
                metadata={
                    **trace_metadata,
                    "final_status": final_status,
                    "decision_type": decision_type,
                    "safety_level": refusal_safety,
                    "source_count": 0,
                    "citation_count": 0,
                    "unknown_section_heading_count": 0,
                    "refusal_reason": refusal,
                },
                tags=[
                    "pdf_qa",
                    "vertex_search",
                    f"status:{final_status}",
                    f"safety:{_safe_tag_value(refusal_safety)}",
                    f"decision:{decision_type}",
                    final_status,
                ],
            )
            safe_update_observation(
                pipeline_span,
                output={"status": final_status, "summary": refusal},
                metadata={
                    "final_status": final_status,
                    "decision_type": decision_type,
                    "safety_level": refusal_safety,
                    "source_count": 0,
                    "citation_count": 0,
                    "unknown_section_heading_count": 0,
                },
            )
            return {
                "status": "success",
                "summary": refusal,
                "safety_level": refusal_safety,
                "constraints": [],
                "sources": [],
                "source_count": 0,
                "grounding_chunks": [],
                "citations": [],
                "insufficient_evidence": False,
                "langfuse_trace_id": root_trace_id,
                "langfuse_observation_id": root_observation_id,
            }

        for current_query in _query_variants(query):
            for serving_config in serving_configs:
                url = f"{DISCOVERY_ENGINE_BASE}/{serving_config}:search"
                all_chunks: list[SearchChunk] = []
                page_token = ""

                with langfuse_span(
                    name="vertex_retrieval",
                    input={
                        "retrieval_query": current_query,
                        "serving_config": serving_config,
                        "page_size": page_size,
                        "max_pages": max_pages,
                    },
                    metadata={"serving_config": serving_config, "page_size": page_size, "max_pages": max_pages},
                    version=settings.app_version,
                ) as retrieval_span:
                    for _ in range(max_pages):
                        payload: dict[str, Any] = {
                            "query": current_query,
                            "pageSize": page_size,
                            "contentSearchSpec": {"searchResultMode": "CHUNKS"},
                        }
                        if page_token:
                            payload["pageToken"] = page_token

                        try:
                            response = session.post(url, json=payload, timeout=60)
                        except RequestException as exc:
                            last_error = f"Search request failed: {exc}"
                            safe_update_observation(
                                retrieval_span,
                                level="ERROR",
                                status_message="search request failed",
                                metadata={"error": str(exc), "error_type": type(exc).__name__, "error_stage": "vertex_retrieval"},
                                output={"error": str(exc)},
                            )
                            break

                        if response.status_code == 404:
                            last_error = f"Serving config not found: {serving_config}"
                            safe_update_observation(
                                retrieval_span,
                                level="ERROR",
                                status_message="serving config not found",
                                metadata={"error": last_error, "error_type": "NotFound", "error_stage": "vertex_retrieval"},
                                output={"error": last_error},
                            )
                            break
                        if response.status_code >= 400:
                            last_error = f"Search failed ({response.status_code}): {response.text}"
                            safe_update_observation(
                                retrieval_span,
                                level="ERROR",
                                status_message="search failed",
                                metadata={"error": last_error, "error_type": "HttpError", "status_code": response.status_code, "error_stage": "vertex_retrieval"},
                                output={"error": last_error},
                            )
                            break

                        try:
                            data = response.json()
                        except ValueError:
                            last_error = "Search returned invalid JSON response"
                            safe_update_observation(
                                retrieval_span,
                                level="ERROR",
                                status_message="invalid json response",
                                metadata={"error": last_error, "error_type": "InvalidJson", "error_stage": "vertex_retrieval"},
                                output={"error": last_error},
                            )
                            break

                        raw_results = data.get("results", [])
                        if not isinstance(raw_results, list):
                            raw_results = []

                        normalized_chunks = _normalize_search_results(raw_results)
                        all_chunks.extend(normalized_chunks)

                        retrieval_preview = []
                        for chunk in normalized_chunks[:5]:
                            retrieval_preview.append(
                                {
                                    "uri": chunk.uri,
                                    "section_heading": chunk.title,
                                    "page": chunk.metadata.get("page"),
                                    "preview": compact_text(chunk.text, 180),
                                }
                            )

                        next_page_token = data.get("nextPageToken", "")
                        retrieval_summary = {
                            "search_status": "success",
                            "raw_result_count": len(raw_results),
                            "normalized_chunk_count": len(normalized_chunks),
                            "total_chunk_count": len(all_chunks),
                            "result_ids": [str(chunk.get("chunk", {}).get("name", "")) for chunk in raw_results[:10] if isinstance(chunk, dict)],
                            "section_headings": [chunk.title for chunk in normalized_chunks[:10]],
                            "page_numbers": [chunk.metadata.get("page") for chunk in normalized_chunks[:10]],
                            "preview": retrieval_preview,
                            "next_page_token_present": bool(isinstance(next_page_token, str) and next_page_token),
                        }
                        safe_update_observation(retrieval_span, output=truncate_value(retrieval_summary), metadata=truncate_value(retrieval_summary))

                        if isinstance(next_page_token, str) and next_page_token:
                            page_token = next_page_token
                            continue
                        break

                if not all_chunks:
                    continue

                chunks_with_text = [chunk for chunk in all_chunks if chunk.text.strip()]
                if not chunks_with_text:
                    last_error = (
                        "Search returned results without chunk text. "
                        "Verify datastore chunking/indexing settings."
                    )
                    safe_update_trace(
                        metadata={
                            **trace_metadata,
                            "final_status": "no_evidence",
                            "decision_type": decision_type,
                            "error": last_error,
                            "error_stage": "normalize_evidence",
                            "source_count": 0,
                            "citation_count": 0,
                            "unknown_section_heading_count": 0,
                        },
                        output={"status": "no_evidence", "error": last_error},
                        tags=["pdf_qa", "vertex_search", "status:no_evidence", "no_evidence", "error"],
                    )
                    continue

                with langfuse_span(
                    name="normalize_evidence",
                    input={
                        "chunk_count": len(chunks_with_text),
                        "query": compact_text(query, 240),
                        "max_evidence": max_evidence,
                    },
                    metadata={"chunk_count": len(chunks_with_text), "max_evidence": max_evidence},
                    version=settings.app_version,
                ) as evidence_span:
                    ranked_chunks = _rank_chunks(query, chunks_with_text)
                    evidence_pool = _collect_evidence(query, ranked_chunks)
                    selected_evidence = _select_evidence(query, evidence_pool)[:max_evidence]
                    evidence_summary = {
                        "deduped_evidence_count": len(evidence_pool),
                        "selected_evidence_count": len(selected_evidence),
                        "chunk_ids": [e.chunk.document_id for e in selected_evidence],
                        "section_headings": [e.chunk.title for e in selected_evidence],
                        "pages": [e.chunk.metadata.get("page") for e in selected_evidence],
                        "anchor_hits": [e.anchor_hits for e in selected_evidence],
                        "number_overlap": [e.overlap for e in selected_evidence],
                        "table_like_content": _is_likely_table_query(query),
                        "evidence_sufficient": bool(selected_evidence),
                    }
                    safe_update_observation(evidence_span, output=truncate_value(evidence_summary), metadata=truncate_value(evidence_summary))

                prompt_system_summary = (
                    "Answer only from the supplied evidence. Preserve exact numbers, units, thresholds, conditions, and procedural order. "
                    "If the evidence is insufficient, say so explicitly. Do not invent facts or cite unrelated chunks."
                )
                prompt_payload = {
                    "template_name": "evidence_grounded_pdf_qa_v1",
                    "template_version": "1.0",
                    "system_instruction_summary": prompt_system_summary,
                    "query": query,
                    "evidence_count": len(selected_evidence),
                    "evidence_ids": [e.chunk.uri for e in selected_evidence],
                    "evidence_preview": [
                        {
                            "uri": e.chunk.uri,
                            "section_heading": e.chunk.title,
                            "page": e.chunk.metadata.get("page"),
                            "snippet": compact_text(e.sentence, 220),
                        }
                        for e in selected_evidence
                    ],
                }
                if settings.capture_full_prompt:
                    prompt_payload["prompt_text"] = "\n".join(
                        [
                            "SYSTEM:",
                            prompt_system_summary,
                            "",
                            f"QUESTION: {query}",
                            "",
                            "EVIDENCE:",
                            *[
                                f"- {e.chunk.uri} | {e.chunk.title} | page={e.chunk.metadata.get('page')} | {e.sentence}"
                                for e in selected_evidence
                            ],
                        ]
                    )

                with langfuse_span(
                    name="build_prompt",
                    input=truncate_value(prompt_payload),
                    metadata={
                        "template_name": prompt_payload["template_name"],
                        "template_version": prompt_payload["template_version"],
                        "evidence_count": len(selected_evidence),
                        "prompt_size_chars": len(str(prompt_payload.get("prompt_text", prompt_system_summary))),
                    },
                    version=settings.app_version,
                ) as prompt_span:
                    safe_update_observation(
                        prompt_span,
                        output=truncate_value(
                            {
                                "system_instruction_summary": prompt_system_summary,
                                "evidence_count": len(selected_evidence),
                                "evidence_ids": [e.chunk.uri for e in selected_evidence],
                                "prompt_size_chars": len(str(prompt_payload.get("prompt_text", prompt_system_summary))),
                            }
                        ),
                    )

                try:
                    with langfuse_generation(
                        name="generate_answer",
                        input=truncate_value(prompt_payload),
                        metadata={
                            "model": "deterministic-evidence-synthesizer",
                            "model_parameters": {"temperature": 0, "top_p": 1, "max_tokens": 0},
                            "evidence_count": len(selected_evidence),
                        },
                        model="deterministic-evidence-synthesizer",
                        model_parameters={"temperature": 0, "top_p": 1, "max_tokens": 0},
                        version=settings.app_version,
                    ) as generation_span:
                        assembled = _assemble_answer(query, selected_evidence)
                        safe_update_observation(generation_span, output=truncate_value(assembled), metadata=truncate_value(assembled))
                except Exception as exc:  # noqa: BLE001
                    last_error = f"Answer assembly failed: {type(exc).__name__}: {exc}"
                    safe_update_trace(
                        output=truncate_value({"status": "error", "error": last_error}),
                        metadata={
                            **trace_metadata,
                            "final_status": "error",
                            "decision_type": decision_type,
                            "error": last_error,
                            "error_type": type(exc).__name__,
                            "error_stage": "generate_answer",
                            "source_count": 0,
                            "citation_count": 0,
                            "unknown_section_heading_count": 0,
                        },
                        tags=["pdf_qa", "vertex_search", "status:error", "search_status:error", "error"],
                    )
                    safe_update_observation(
                        pipeline_span,
                        level="ERROR",
                        status_message="answer assembly failed",
                        output=truncate_value({"status": "error", "error": last_error}),
                        metadata={
                            "final_status": "error",
                            "error_stage": "generate_answer",
                            "error_type": type(exc).__name__,
                        },
                    )
                    return {
                        "status": "error",
                        "error": last_error,
                        "summary": "",
                        "sources": [],
                        "source_count": 0,
                        "grounding_chunks": [],
                        "citations": [],
                        "insufficient_evidence": True,
                        "langfuse_trace_id": root_trace_id,
                        "langfuse_observation_id": root_observation_id,
                    }

                with langfuse_span(
                    name="parse_model_output",
                    input={"generated_answer_preview": compact_text(str(assembled.get("answer_text", "")), 240)},
                    metadata={"safety_level": assembled.get("safety_level"), "insufficient_evidence": assembled.get("insufficient_evidence", False)},
                    version=settings.app_version,
                ) as parse_span:
                    summary = str(assembled.get("answer_text", "")).strip()
                    if not summary:
                        summary = (
                            "I could not find enough grounded evidence in the handbook to answer this reliably. "
                            "Please refine the question using handbook terms (equipment name, procedure name, threshold, or section context)."
                        )
                    safe_update_observation(
                        parse_span,
                        output={
                            "summary": compact_text(summary, 240),
                            "safety_level": assembled.get("safety_level"),
                            "constraints": assembled.get("constraints", []),
                            "insufficient_evidence": bool(assembled.get("insufficient_evidence", False)),
                        },
                    )

                with langfuse_span(
                    name="format_citations",
                    input={"selected_evidence_count": len(selected_evidence)},
                    metadata={"selected_evidence_count": len(selected_evidence)},
                    version=settings.app_version,
                ) as citations_span:
                    citations = _build_citations(selected_evidence)
                    sources = [{"title": c.get("doc_name", ""), "uri": c.get("uri", "")} for c in citations]
                    citation_summary = _citation_trace_summary(citations)
                    safe_update_observation(citations_span, output=truncate_value(citation_summary), metadata=truncate_value(citation_summary))

                safety_level = str(assembled.get("safety_level") or "unknown")
                unknown_section_heading_count = int(citation_summary.get("unknown_section_heading_count", 0) or 0)

                grounding_chunks = [
                    {
                        "text": chunk.text,
                        "title": chunk.title,
                        "uri": chunk.uri,
                        "metadata": chunk.metadata,
                        "rank_score": chunk.metadata.get("rank_score", 0),
                    }
                    for chunk in ranked_chunks[:12]
                ]

                with langfuse_span(
                    name="assemble_response",
                    input={
                        "summary_preview": compact_text(summary, 240),
                        "source_count": len(sources),
                        "citation_count": len(citations),
                    },
                    metadata={
                        "source_count": len(sources),
                        "citation_count": len(citations),
                        "final_status": "success",
                    },
                    version=settings.app_version,
                ) as response_span:
                    result = {
                        "status": "success",
                        "summary": summary,
                        "safety_level": assembled.get("safety_level"),
                        "constraints": assembled.get("constraints", []),
                        "sources": sources,
                        "source_count": len(sources),
                        "grounding_chunks": grounding_chunks,
                        "citations": citations,
                        "insufficient_evidence": bool(assembled.get("insufficient_evidence", False)),
                        "langfuse_trace_id": root_trace_id,
                        "langfuse_observation_id": root_observation_id,
                    }
                    safe_update_observation(response_span, output=truncate_value(result), metadata={"final_status": "success"})

                final_status = "success"
                success_tags = [
                    "pdf_qa",
                    "vertex_search",
                    "status:success",
                    "search_status:success",
                    "procedural_question" if procedural_query else "",
                    "synthesis_question" if synthesis_query else "",
                    "table_question" if table_query else "",
                    "numeric_question" if numeric_query else "",
                    f"safety:{_safe_tag_value(safety_level)}",
                    f"decision:{decision_type}",
                    "source_count:0" if len(sources) == 0 else "source_count:gt0",
                    "insufficient_evidence" if bool(assembled.get("insufficient_evidence", False)) else "sufficient_evidence",
                ]
                safe_update_trace(
                    output=truncate_value({"status": final_status, "source_count": len(sources), "citation_count": len(citations)}),
                    metadata={
                        **trace_metadata,
                        "final_status": final_status,
                        "decision_type": decision_type,
                        "source_count": len(sources),
                        "citation_count": len(citations),
                        "unknown_section_heading_count": unknown_section_heading_count,
                        "safety_level": safety_level,
                    },
                    tags=[tag for tag in success_tags if tag],
                )
                safe_update_observation(
                    pipeline_span,
                    output=truncate_value({"status": final_status, "source_count": len(sources), "citation_count": len(citations)}),
                    metadata={
                        "final_status": final_status,
                        "decision_type": decision_type,
                        "safety_level": safety_level,
                        "source_count": len(sources),
                        "citation_count": len(citations),
                        "unknown_section_heading_count": unknown_section_heading_count,
                    },
                )
                return result

        final_status = "no_evidence" if last_error and "Search returned results without chunk text" in last_error else "error"
        error_stage = "normalize_evidence" if final_status == "no_evidence" else "vertex_retrieval"
        safe_update_trace(
            output=truncate_value({"status": final_status, "error": last_error or "Unknown Discovery Engine search error"}),
            metadata={
                **trace_metadata,
                "final_status": final_status,
                "decision_type": decision_type,
                "error": last_error or "Unknown Discovery Engine search error",
                "error_stage": error_stage,
                "source_count": 0,
                "citation_count": 0,
                "unknown_section_heading_count": 0,
            },
            tags=["pdf_qa", "vertex_search", f"status:{final_status}", "search_status:error", "error"],
        )
        safe_update_observation(
            pipeline_span,
            output=truncate_value({"status": final_status, "error": last_error or "Unknown Discovery Engine search error"}),
            metadata={
                "final_status": final_status,
                "decision_type": decision_type,
                "error_stage": error_stage,
                "source_count": 0,
                "citation_count": 0,
                "unknown_section_heading_count": 0,
            },
        )
        return {
            "status": "error",
            "error": last_error or "Unknown Discovery Engine search error",
            "summary": "",
            "sources": [],
            "source_count": 0,
            "grounding_chunks": [],
            "citations": [],
            "insufficient_evidence": True,
            "langfuse_trace_id": root_trace_id,
            "langfuse_observation_id": root_observation_id,
        }


def print_search_results(result: dict[str, Any]) -> None:
    status = str(result.get("status", "unknown"))

    status_icon = "[green]OK[/green]" if status == "success" else "[red]ERROR[/red]"
    CONSOLE.print(f"{status_icon} Status: {status}")

    if status == "error":
        error = str(result.get("error", "Unknown error"))
        CONSOLE.print(Panel(error, title="Error", border_style="red"))
        return

    summary = str(result.get("summary", ""))
    citations = result.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    annotated_summary, used_citations = _annotate_summary_with_citations(summary, citations)

    if annotated_summary:
        CONSOLE.print(
            Panel(
                annotated_summary,
                title="[bold cyan]Answer[/bold cyan]",
                border_style="cyan",
                expand=True,
            )
        )
    else:
        CONSOLE.print(
            Panel(
                "[dim]No summary available[/dim]",
                title="[bold cyan]Answer[/bold cyan]",
                border_style="cyan",
                expand=True,
            )
        )

    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    source_count = int(result.get("source_count", 0))

    unique_sources: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    for src in sources:
        if not isinstance(src, dict):
            continue
        uri = str(src.get("uri", ""))
        if uri and uri not in seen_uris:
            seen_uris.add(uri)
            unique_sources.append(src)

    if unique_sources:
        src_table = Table(
            title=f"Unique Sources ({len(unique_sources)} unique, {source_count} total citations)",
            show_header=True,
        )
        src_table.add_column("#", style="dim", width=3)
        src_table.add_column("Title", style="cyan", width=40)
        src_table.add_column("URI", style="dim", width=50)

        for i, src in enumerate(unique_sources, 1):
            title = str(src.get("title", ""))
            uri = str(src.get("uri", ""))
            src_table.add_row(
                str(i),
                title[:38] + "..." if len(title) > 38 else title,
                uri[:47] + "..." if len(uri) > 47 else uri,
            )

        CONSOLE.print(src_table)
    else:
        CONSOLE.print("[dim]No sources retrieved[/dim]")

    CONSOLE.print("\n[dim]Response metadata:[/dim]")
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Key", style="dim")
    meta_table.add_column("Value", style="white")
    meta_table.add_row("Unique sources", str(len(unique_sources)))
    meta_table.add_row("Total citations", str(source_count))
    meta_table.add_row("Structured citations", str(len(used_citations)))
    meta_table.add_row("Insufficient evidence", str(bool(result.get("insufficient_evidence", False))))
    CONSOLE.print(meta_table)

    if used_citations:
        CONSOLE.print("\n[bold cyan]Citations[/bold cyan]")
        for i, citation in enumerate(used_citations, 1):
            CONSOLE.print(f"[{i}]")
            CONSOLE.print(f"Doc Name: {citation.get('doc_name', 'Unknown')} ({citation.get('doc_name_source', 'unknown source')})")
            CONSOLE.print(
                f"Section Title: {citation.get('section_title', 'Unknown')} "
                f"({citation.get('section_title_source', 'unknown source')})"
            )
            CONSOLE.print(
                f"Page number: {citation.get('page_number', 'Unknown')} "
                f"({citation.get('page_number_source', 'unknown source')})"
            )
            CONSOLE.print(
                f"Number of the paragraph: {citation.get('paragraph_number', 'Unknown')} "
                f"({citation.get('paragraph_number_source', 'unknown source')})"
            )
            CONSOLE.print(f"Support score: {citation.get('support_score', 'Unknown')}")
            CONSOLE.print(f"Short paragraph citation (max 50 words): {citation.get('snippet', 'Unknown')}")
            CONSOLE.print()


def _resolve_query(arg_query: str | None) -> str:
    if arg_query and arg_query.strip():
        return arg_query.strip()

    env_query = os.getenv("VERTEX_QA_QUERY", "").strip()
    if env_query:
        return env_query

    raise ValueError("Query is required. Pass --query or set VERTEX_QA_QUERY.")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask a Vertex AI datastore-backed PDF QA question.")
    parser.add_argument("--query", type=str, help="Question to answer")
    args = parser.parse_args()

    try:
        query = _resolve_query(args.query)
    except ValueError as exc:
        CONSOLE.print(f"[red]ERROR[/red] {exc}")
        return 2

    try:
        credentials = get_credentials()
        config = Configs()  # type: ignore[call-arg]
    except Exception as exc:
        CONSOLE.print(Panel(str(exc), title="Configuration Error", border_style="red"))
        return 2

    if not config.vertex_datastore_id:
        CONSOLE.print("[red]ERROR[/red] VERTEX_AI_DATASTORE_ID is not set.")
        CONSOLE.print("[dim]Set it in your .env file and rerun.[/dim]")
        return 2

    cfg_table = Table(title="Vertex AI Search Configuration", show_header=False)
    cfg_table.add_column("Key", style="cyan")
    cfg_table.add_column("Value", style="white")
    cfg_table.add_row("Data store", config.vertex_datastore_id)
    cfg_table.add_row("Region", str(config.google_cloud_location))
    cfg_table.add_row("Query", query[:120] + ("..." if len(query) > 120 else ""))
    cfg_table.add_row("Page size", str(_get_int_env("VERTEX_SEARCH_PAGE_SIZE", DEFAULT_PAGE_SIZE)))
    cfg_table.add_row("Max pages", str(_get_int_env("VERTEX_SEARCH_MAX_PAGES", DEFAULT_MAX_PAGES)))
    cfg_table.add_row("Max evidence sentences", str(_get_int_env("VERTEX_MAX_EVIDENCE_SENTENCES", DEFAULT_MAX_EVIDENCE_SENTENCES)))
    cfg_table.add_row("Model", "N/A (deterministic evidence-grounded synthesis)")
    cfg_table.add_row("Auth", "Attached GCE service account or ADC")
    CONSOLE.print(cfg_table)

    result = search_discovery_engine(query, config, credentials)

    CONSOLE.print()
    CONSOLE.print("[bold]Search Results:[/bold]")
    CONSOLE.print()
    print_search_results(result)

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
