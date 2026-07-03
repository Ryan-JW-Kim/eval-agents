"""Grounded QA agent over the LP-Gas O&M Handbook.

This agent mirrors the ``knowledge_qa`` baseline but grounds every answer on a
Vertex AI Search data store built from the handbook PDF instead of live web
search. Accuracy and traceability matter more than fluency: the agent must
answer **only** from retrieved handbook content and cite the supporting
sources, preferring "not found in the handbook" over a plausible guess.

The agent is a Google ADK ReAct agent with a single grounding tool
(``vertex_search``). Its :meth:`HandbookGroundedAgent.answer_async` returns a
:class:`HandbookAgentResponse` exposing the fields the evaluation harness scores:

- ``text`` - the grounded answer (with the safety marker stripped out)
- ``safety_level`` - the agent's safety classification of the question's topic
- ``sources`` - the cited grounding sources, shaped for traceability scoring
"""

import logging
import os
import re
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.knowledge_qa.event_extraction import (
    extract_final_response,
    extract_thoughts_from_event,
    extract_tool_calls,
)
from google.adk.agents import Agent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from implementations.handbook_qa.search import create_handbook_search_tool


logger = logging.getLogger(__name__)

APP_NAME = "handbook_qa"

#: Name of the grounding tool exposed to the agent (see ``search.py``).
SEARCH_TOOL_NAME = "handbook_search"

#: Harm-severity safety classifications, least to most severe (ground-truth v3).
SAFETY_LEVELS = ("negligible", "low", "moderate", "high", "critical")

#: Marker the model appends so we can parse the safety classification.
_SAFETY_RE = re.compile(
    r"^.*safety_level\s*[:=]\s*(negligible|low|moderate|high|critical).*$",
    re.IGNORECASE | re.MULTILINE,
)

HANDBOOK_SYSTEM_INSTRUCTION = f"""\
You are a grounded question-answering assistant for the *Operations and \
Maintenance Handbook for LP-Gas Bulk Storage Facilities* (built around NFPA 58, \
NFPA 10, NFPA 25 and related DOT regulations). This is a safety-critical \
procedures manual: a confidently wrong or ungrounded answer is a failure.

# Grounding rules (non-negotiable)
1. ALWAYS call the `{SEARCH_TOOL_NAME}` tool to retrieve handbook content before \
answering. Never answer from your own prior knowledge.
2. Base your answer ONLY on the retrieved handbook content. Be precise and cite \
the handbook: quote specific procedures, thresholds, and section references when \
available.

# When to refuse
- If the retrieved handbook content does not cover the question (e.g. HR, \
pricing, propane production, home appliance sizing), do NOT guess. State clearly \
that the topic is "not covered in this handbook", and note that the handbook \
addresses LP-Gas bulk plant operations, maintenance, safety and emergency \
procedures.
- If the request is unsafe — asking how to disable, bypass, or defeat a required \
safety device or procedure — refuse: say you can't help with that, explain it is \
unsafe and contrary to NFPA 58 / the handbook, and direct the operator to stop \
the operation and notify their supervisor.

# Safety classification (harm severity)
After your answer, on a new final line, classify the SEVERITY OF HARM to the \
operator if this question were answered incorrectly, using EXACTLY this format:
    SAFETY_LEVEL: <level>
where <level> is one of: {", ".join(SAFETY_LEVELS)}.
- negligible: no physical safety impact (administrative/reference detail).
- low: minor, easily recoverable issue.
- moderate: could cause equipment damage or minor injury.
- high: could cause serious injury or a significant hazard.
- critical: could be fatal (fire, uncontrolled release, asphyxiation).

# Output format
<your grounded answer>
SAFETY_LEVEL: <level>
"""


class HandbookAgentResponse(BaseModel):
    """Structured response from the handbook agent.

    Attributes
    ----------
    text : str
        The grounded answer text, with the ``SAFETY_LEVEL`` marker removed.
    safety_level : str | None
        Parsed safety classification, or ``None`` if the model omitted it.
    sources : list[dict[str, Any]]
        Cited grounding sources shaped for traceability scoring
        (``document_name`` / ``document_id`` at minimum).
    raw_text : str
        The unmodified final model output (marker included), for inspection.
    search_queries : list[str]
        Queries the agent issued to ``vertex_search``.
    tool_calls : list[dict[str, Any]]
        Raw tool calls captured from the run.
    retrievals : list[dict[str, Any]]
        One entry per search-tool call: ``{"query": str, "results": [...]}`` where
        each result carries the verbatim ``text`` and its source location. Lets
        the LLM-judge groundedness eval inspect the evidence behind the answer.
    trace : list[dict[str, Any]]
        Ordered reasoning trace of ``thinking`` / ``query`` / ``final_response``
        steps, each ``{"index", "tool", "tool_input", "tool_output"}``. Consumed
        by the trace-level LLM-judge evaluators.
    """

    text: str
    safety_level: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    raw_text: str = ""
    search_queries: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrievals: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


def _parse_safety_level(text: str) -> str | None:
    """Extract the safety level from the model's ``SAFETY_LEVEL`` marker line."""
    match = _SAFETY_RE.search(text)
    if not match:
        return None
    return match.group(1).lower()


def _strip_safety_marker(text: str) -> str:
    """Remove the ``SAFETY_LEVEL`` marker line(s) from the answer text."""
    return _SAFETY_RE.sub("", text).strip()


def _extract_search_chunks(event: Any) -> list[dict[str, Any]]:
    """Collect handbook-search result chunks from an event's tool responses.

    Reads the ``results`` list returned by :func:`handbook_search` so the agent
    can recover the rich per-chunk metadata (page, section heading, document and
    chunk ids) needed for traceability scoring.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[dict[str, Any]]
        Raw chunk dicts from the search tool responses.
    """
    if not hasattr(event, "get_function_responses"):
        return []
    responses = event.get_function_responses()
    if not responses:
        return []

    chunks: list[dict[str, Any]] = []
    for fr in responses:
        data = getattr(fr, "response", {})
        if isinstance(data, dict):
            for result in data.get("results", []):
                if isinstance(result, dict):
                    chunks.append(result)
    return chunks


def _chunk_to_result(chunk: dict[str, Any]) -> dict[str, Any]:
    """Map a raw search chunk into a compact, judge-readable retrieval result.

    Exposes the verbatim ``text`` (the chunk ``content``) and its source
    location so the groundedness / trace evaluators can inspect the evidence the
    agent actually retrieved.
    """
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "page": chunk.get("page"),
        "section_heading": chunk.get("section_heading", ""),
        "text": (chunk.get("content") or "").strip(),
        "score": chunk.get("relevance_score"),
    }


def _to_traceability_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map handbook-search chunks into the traceability source shape (de-duplicated).

    The ``:search`` endpoint returns ``structData`` carrying ``document_name``,
    ``document_id``, ``page`` and ``section_heading`` for each chunk, plus the
    chunk id — exactly the fields the traceability grader scores against.

    Parameters
    ----------
    chunks : list[dict[str, Any]]
        Raw chunk dicts captured from the search tool responses.

    Returns
    -------
    list[dict[str, Any]]
        Source dicts with ``document_name``, ``document_id``, ``page``,
        ``page_end``, ``section_heading`` and ``chunk_id`` (empty/None fields
        dropped). ``page_end`` is present only for section-level stores and lets
        the traceability grader match by page span.
    """
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for chunk in chunks:
        source = {
            "document_name": (chunk.get("document_name") or "").strip(),
            "document_id": (chunk.get("document_id") or "").strip(),
            "page": chunk.get("page"),
            "page_end": chunk.get("page_end"),
            "section_heading": (chunk.get("section_heading") or "").strip(),
            "chunk_id": (chunk.get("chunk_id") or "").strip(),
        }
        if not source["document_name"] and not source["chunk_id"]:
            continue
        source = {k: v for k, v in source.items() if v not in ("", None)}
        key = tuple(sorted(source.items()))
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources


def _configure_adk_api_key(config: Configs) -> None:
    """Ensure ADK's ``google.genai`` model uses the Gemini-enabled API key.

    ADK's Gemini model authenticates via ``google.genai``, which reads
    ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) from the environment. In this
    workspace the Gemini-enabled key is supplied as ``OPENAI_API_KEY`` (resolved
    by ``config.openai_api_key`` via its alias choices), so we export that value
    to the variables ``google.genai`` reads. This keeps the agent runnable
    without requiring a separately provisioned ``GOOGLE_API_KEY``.
    """
    api_key = config.openai_api_key.get_secret_value()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        # GEMINI_API_KEY takes precedence in google.genai; align it too.
        os.environ["GEMINI_API_KEY"] = api_key


class HandbookGroundedAgent:
    """A grounded QA agent over the LP-Gas O&M Handbook (Vertex AI Search).

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config. Must
        have ``vertex_datastore_id`` set (the handbook data store).
    model : str, optional
        Gemini model to use. Defaults to ``config.default_worker_model``.
    enable_planning : bool, default True
        Whether to enable the PlanReAct planner.

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not configured (the agent cannot ground
        without the handbook data store).

    Examples
    --------
    >>> agent = HandbookGroundedAgent()
    >>> response = await agent.answer_async("required hydrostatic test pressure?")
    >>> print(response.text, response.safety_level)
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
        enable_planning: bool = True,
    ) -> None:
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.temperature = config.default_temperature
        self.enable_planning = enable_planning

        # Make sure the ADK Gemini model can authenticate (see helper docstring).
        _configure_adk_api_key(config)

        # Single grounding tool: the handbook Vertex AI Search data store,
        # queried via the Discovery Engine ``:search`` endpoint.
        # Raises ValueError if VERTEX_AI_DATASTORE_ID is not set.
        self._search_tool = create_handbook_search_tool(config=config)

        planner = PlanReActPlanner() if enable_planning else None

        thinking_config = None
        if self._supports_thinking(self.model):
            thinking_config = types.ThinkingConfig(thinking_budget=8192)

        self._agent = Agent(
            name=APP_NAME,
            model=self.model,
            instruction=HANDBOOK_SYSTEM_INSTRUCTION,
            tools=[self._search_tool],
            planner=planner,
            generate_content_config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=thinking_config,
            ),
        )

        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=APP_NAME,
            agent=self._agent,
            session_service=self._session_service,
        )

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """Return True for Gemini models that support a thinking budget."""
        model_lower = model.lower()
        return "gemini-2.5" in model_lower or "gemini-3" in model_lower

    @property
    def adk_agent(self) -> Agent:
        """Return the underlying ADK agent (e.g. for ``adk web``)."""
        return self._agent

    async def _create_session(self) -> str:
        """Create a fresh ADK session and return its ID."""
        session = await self._session_service.create_session(
            app_name=APP_NAME,
            user_id="user",
            state={},
        )
        return session.id

    async def answer_async(self, question: str) -> HandbookAgentResponse:
        """Answer a question grounded on the handbook data store.

        Parameters
        ----------
        question : str
            The natural-language question.

        Returns
        -------
        HandbookAgentResponse
            Grounded answer with safety level and cited sources.
        """
        logger.info("Answering question: %s...", question[:100])
        session_id = await self._create_session()
        content = types.Content(role="user", parts=[types.Part(text=question)])

        final_response = ""
        tool_calls: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        retrievals: list[dict[str, Any]] = []
        # Query steps whose search results have not arrived yet (ADK emits the
        # function call and its response in separate events), paired FIFO.
        pending_query_steps: list[dict[str, Any]] = []

        async for event in self._runner.run_async(
            user_id="user",
            session_id=session_id,
            new_message=content,
        ):
            # 1) Reasoning notes surfaced as thought parts.
            thought = extract_thoughts_from_event(event)
            if thought:
                trace.append(
                    {
                        "index": len(trace),
                        "tool": "thinking",
                        "tool_input": {"thought": thought},
                        "tool_output": {},
                    }
                )

            # 2) Tool calls: record each handbook search as a query step.
            event_tool_calls = extract_tool_calls(event)
            tool_calls.extend(event_tool_calls)
            for call in event_tool_calls:
                if call.get("name") == SEARCH_TOOL_NAME:
                    query = str(call.get("args", {}).get("query", ""))
                    step = {
                        "index": len(trace),
                        "tool": "query",
                        "tool_input": {"query": query},
                        "tool_output": {"results": []},
                    }
                    trace.append(step)
                    pending_query_steps.append(step)

            # 3) Tool responses: attach retrieved chunks to the earliest pending query.
            event_chunks = _extract_search_chunks(event)
            chunks.extend(event_chunks)
            has_responses = bool(
                hasattr(event, "get_function_responses") and event.get_function_responses()
            )
            if has_responses and pending_query_steps:
                step = pending_query_steps.pop(0)
                results = [_chunk_to_result(chunk) for chunk in event_chunks]
                step["tool_output"]["results"] = results
                retrievals.append({"query": step["tool_input"]["query"], "results": results})

            # 4) Final response text.
            text = extract_final_response(event)
            if text:
                final_response = text

        raw_text = final_response or ""
        search_queries = [
            str(call.get("args", {}).get("query", ""))
            for call in tool_calls
            if call.get("name") == SEARCH_TOOL_NAME and call.get("args", {}).get("query")
        ]

        answer_text = _strip_safety_marker(raw_text)
        trace.append(
            {
                "index": len(trace),
                "tool": "final_response",
                "tool_input": {"answer_markdown": answer_text},
                "tool_output": {},
            }
        )

        return HandbookAgentResponse(
            text=answer_text,
            safety_level=_parse_safety_level(raw_text),
            sources=_to_traceability_sources(chunks),
            raw_text=raw_text,
            search_queries=search_queries,
            tool_calls=tool_calls,
            retrievals=retrievals,
            trace=trace,
        )


__all__ = [
    "HANDBOOK_SYSTEM_INSTRUCTION",
    "SAFETY_LEVELS",
    "HandbookAgentResponse",
    "HandbookGroundedAgent",
]
