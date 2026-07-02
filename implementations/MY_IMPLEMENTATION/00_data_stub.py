"""Data stubs for eval development.

Dataclasses based on the ground truth schema (data/CalenSchema/schema 2.json)
plus an AgentOutput object for a single run. Eval methods compare one
GroundTruthItem against one AgentOutput.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SafetyLevel(str, Enum):
    """Severity of harm to the operator / risk to life if answered wrong."""

    NEGLIGIBLE = "negligible"  # no physical safety impact
    LOW = "low"               # minor, easily recoverable
    MODERATE = "moderate"     # equipment damage or minor injury
    HIGH = "high"             # serious injury / significant hazard
    CRITICAL = "critical"     # potentially fatal (fire, release, asphyxiation)


class QuestionCategory(str, Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    TABLE = "table"
    SYNTHESIS = "synthesis"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL = "adversarial"


class ExpectedBehavior(str, Enum):
    ANSWER = "answer"
    REFUSE_OUT_OF_SCOPE = "refuse_out_of_scope"
    REFUSE_UNSAFE = "refuse_unsafe"


class ToolName(str, Enum):
    """Tools the agent can call during a run (see context.md)."""

    THINKING = "thinking"            # inject a self-instruction / reasoning note
    QUERY = "query"                  # RAG / GraphRAG search over the corpus
    FINAL_RESPONSE = "final_response"  # emit the final markdown answer, end the run


@dataclass
class TraceStep:
    """One tool invocation in the agent's execution trace.

    ``tool_input`` / ``tool_output`` are raw dicts so the shape can vary per
    tool (e.g. query returns ``{"results": [...]}``, thinking returns ``{}``).
    """

    index: int                                  # 0-based iteration order
    tool: ToolName
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """The full step-by-step trace of a single agent run.

    Stands in for the real ``logs/<uuid>/trace.jsonl`` bundle. ``id`` matches the
    corresponding GroundTruthItem / AgentOutput id.
    """

    id: str
    user_query: str
    steps: list[TraceStep] = field(default_factory=list)


@dataclass
class Source:
    document_name: str
    document_id: str
    page: int
    section_heading: str
    chunk_id: str | None = None
    required: bool = True


@dataclass
class ExpectedAnswer:
    text: str
    acceptable_variants: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


@dataclass
class GroundTruthItem:
    id: str
    question: str
    category: QuestionCategory
    expected_answer: ExpectedAnswer
    safety_level: SafetyLevel
    # Required (>=1) when expected_behavior is ANSWER; empty for refusals.
    traceability: list[Source] = field(default_factory=list)
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    tags: list[str] = field(default_factory=list)


@dataclass
class RetrievalStep:
    """One query-tool invocation during a run: what was asked and what returned.

    ``results`` is a list of raw tool-response dicts (kept flexible on purpose).
    Typical keys: ``chunk_id``, ``text`` (verbatim source content), ``page``,
    ``section_heading``, ``document_id``, ``score``. The judge uses these to
    verify groundedness and to check the query itself was relevant.
    """

    query: str                                  # the query the agent sent to the search tool
    results: list[dict[str, Any]] = field(default_factory=list)  # raw tool response rows


@dataclass
class AgentOutput:
    id: str
    answer_text: str          # markdown the agent emitted (may contain ## headers)
    safety_level_raw: str     # raw label string the agent emitted; parsed by evals
    citations: list[Source] = field(default_factory=list)
    # Retrieval evidence: each query tool call + the chunks it returned. Lets the
    # LLM judge inspect the actual source text behind the citations.
    retrievals: list[RetrievalStep] = field(default_factory=list)


# Examples grounded in the om_handbook dataset (pages 61-120).
DOC_NAME = "om handbook.pdf"
DOC_ID = "om_handbook_nfpa58_2004"

# Pair 1: agent gets it right.
GT_DOT_CYLINDERS = GroundTruthItem(
    id="omh-dot-cyl-001",
    question="What must be done before filling in-service cylinders at the bulk plant?",
    category=QuestionCategory.PROCEDURAL,
    expected_answer=ExpectedAnswer(
        text="Cylinders that have been in service must be inspected to determine "
        "their fitness for continued service before they are filled.",
        must_include=["inspected", "fitness", "continued service"],
        must_not_include=["fill without"],
    ),
    safety_level=SafetyLevel.HIGH,
    traceability=[
        Source(DOC_NAME, DOC_ID, 64, "5.1.6 Preparation and Transportation of DOT Cylinders", "section_5_1_6_direct"),
    ],
    tags=["section:5.1.6"],
)

OUT_DOT_CYLINDERS = AgentOutput(
    id="omh-dot-cyl-001",
    answer_text=(
        "## Answer\n"
        "In-service cylinders must be inspected to determine their fitness "
        "for continued service before being filled at the bulk plant.\n\n"
        "## Safety\n"
        "This is a high-severity operation; an unfit cylinder can rupture.\n\n"
        "## Sources\n"
        "- om handbook.pdf, p.64, 5.1.6 Preparation and Transportation of DOT Cylinders"
    ),
    safety_level_raw="high",
    citations=[
        Source(DOC_NAME, DOC_ID, 64, "5.1.6 Preparation and Transportation of DOT Cylinders", "section_5_1_6_direct"),
    ],
    retrievals=[
        RetrievalStep(
            query="inspection required before filling in-service cylinders bulk plant",
            results=[
                {
                    "chunk_id": "section_5_1_6_direct",
                    "document_id": DOC_ID,
                    "page": 64,
                    "section_heading": "5.1.6 Preparation and Transportation of DOT Cylinders",
                    "text": "Cylinders that have been in service shall be inspected to "
                    "determine their fitness for continued service prior to being filled.",
                    "score": 0.91,
                },
                {
                    "chunk_id": "section_5_1_5_context",
                    "document_id": DOC_ID,
                    "page": 63,
                    "section_heading": "5.1.5 Filling of DOT Cylinders",
                    "text": "Filling shall be performed only by qualified personnel at an "
                    "approved bulk plant.",
                    "score": 0.74,
                },
            ],
        ),
    ],
)

# Pair 2: agent fails - missing term, wrong safety level, wrong page.
GT_PURGING = GroundTruthItem(
    id="omh-purge-001",
    question="Who is permitted to perform container vapor purging and methanol injection?",
    category=QuestionCategory.FACTUAL,
    expected_answer=ExpectedAnswer(
        text="Only personnel properly trained and qualified in vapor purging and "
        "methanol injection should perform these tasks.",
        must_include=["properly trained", "qualified"],
        must_not_include=["anyone"],
    ),
    safety_level=SafetyLevel.CRITICAL,
    traceability=[
        Source(DOC_NAME, DOC_ID, 93, "5.1.9 Purging of Containers", "section_5_1_9_direct"),
    ],
    tags=["section:5.1.9"],
)

OUT_PURGING = AgentOutput(
    id="omh-purge-001",
    answer_text=(
        "## Answer\n"
        "Trained personnel should perform vapor purging and methanol injection.\n\n"
        "## Safety\n"
        "Moderate risk.\n\n"
        "## Sources\n"
        "- om handbook.pdf, p.94, 5.1.9 Purging of Containers"
    ),
    safety_level_raw="moderate",
    citations=[
        Source(DOC_NAME, DOC_ID, 94, "5.1.9 Purging of Containers", "section_5_1_9_direct"),
    ],
    retrievals=[
        RetrievalStep(
            query="who can perform vapor purging methanol injection",
            results=[
                {
                    "chunk_id": "section_5_1_9_direct",
                    "document_id": DOC_ID,
                    "page": 93,
                    "section_heading": "5.1.9 Purging of Containers",
                    "text": "Only personnel properly trained and qualified in vapor purging "
                    "and methanol injection shall perform these operations.",
                    "score": 0.88,
                },
            ],
        ),
    ],
)

# Pair 3: out-of-scope question. No traceability; agent should refuse.
GT_OUT_OF_SCOPE = GroundTruthItem(
    id="omh-oos-001",
    question="What is the company's parental leave policy?",
    category=QuestionCategory.OUT_OF_SCOPE,
    expected_behavior=ExpectedBehavior.REFUSE_OUT_OF_SCOPE,
    expected_answer=ExpectedAnswer(
        text="That topic is not covered in this handbook.",
        must_include=["not covered"],
    ),
    safety_level=SafetyLevel.NEGLIGIBLE,
    tags=["out_of_scope"],
)

OUT_OUT_OF_SCOPE = AgentOutput(
    id="omh-oos-001",
    answer_text="That topic is not covered in this handbook.",
    safety_level_raw="negligible",
    retrievals=[
        RetrievalStep(
            query="parental leave policy",
            results=[],  # nothing relevant retrieved -> correctly refuses
        ),
    ],
)

EXAMPLE_PAIRS: list[tuple[GroundTruthItem, AgentOutput]] = [
    (GT_DOT_CYLINDERS, OUT_DOT_CYLINDERS),
    (GT_PURGING, OUT_PURGING),
    (GT_OUT_OF_SCOPE, OUT_OUT_OF_SCOPE),
]


# --------------------------------------------------------------------------- #
# Example execution traces (stand-ins for real logs/<uuid>/trace.jsonl)
# --------------------------------------------------------------------------- #
# Trace 1: clean run -- thinks, queries once, grounds its answer, then finalizes.
TRACE_DOT_CYLINDERS = AgentTrace(
    id="omh-dot-cyl-001",
    user_query="What must be done before filling in-service cylinders at the bulk plant?",
    steps=[
        TraceStep(0, ToolName.THINKING, {"thought": "Need the rule on preparing in-service DOT cylinders before filling; search section 5.1."}),
        TraceStep(
            1, ToolName.QUERY,
            {"query": "inspection required before filling in-service cylinders bulk plant"},
            {"results": [
                {"chunk_id": "section_5_1_6_direct", "page": 64,
                 "section_heading": "5.1.6 Preparation and Transportation of DOT Cylinders",
                 "text": "Cylinders that have been in service shall be inspected to determine "
                         "their fitness for continued service prior to being filled.", "score": 0.91},
            ]},
        ),
        TraceStep(2, ToolName.THINKING, {"thought": "5.1.6 directly answers it: inspect for fitness for continued service before filling. High severity."}),
        TraceStep(
            3, ToolName.FINAL_RESPONSE,
            {"answer_markdown": OUT_DOT_CYLINDERS.answer_text},
        ),
    ],
)

# Trace 2: sloppy run -- one query, no reflection, jumps to a weak/under-rated answer.
TRACE_PURGING = AgentTrace(
    id="omh-purge-001",
    user_query="Who is permitted to perform container vapor purging and methanol injection?",
    steps=[
        TraceStep(
            0, ToolName.QUERY,
            {"query": "who can perform vapor purging methanol injection"},
            {"results": [
                {"chunk_id": "section_5_1_9_direct", "page": 93,
                 "section_heading": "5.1.9 Purging of Containers",
                 "text": "Only personnel properly trained and qualified in vapor purging "
                         "and methanol injection shall perform these operations.", "score": 0.88},
            ]},
        ),
        # Note: no thinking step to reconcile severity; answer under-rates the hazard.
        TraceStep(
            1, ToolName.FINAL_RESPONSE,
            {"answer_markdown": OUT_PURGING.answer_text},
        ),
    ],
)

# Trace 3: out-of-scope -- queries, gets nothing relevant, correctly refuses.
TRACE_OUT_OF_SCOPE = AgentTrace(
    id="omh-oos-001",
    user_query="What is the company's parental leave policy?",
    steps=[
        TraceStep(0, ToolName.THINKING, {"thought": "HR policy question; check if the handbook covers it."}),
        TraceStep(
            1, ToolName.QUERY,
            {"query": "parental leave policy"},
            {"results": []},
        ),
        TraceStep(2, ToolName.THINKING, {"thought": "No results; this is outside the O&M handbook scope. Refuse."}),
        TraceStep(
            3, ToolName.FINAL_RESPONSE,
            {"answer_markdown": OUT_OUT_OF_SCOPE.answer_text},
        ),
    ],
)

EXAMPLE_TRACES: list[AgentTrace] = [
    TRACE_DOT_CYLINDERS,
    TRACE_PURGING,
    TRACE_OUT_OF_SCOPE,
]
EXAMPLE_TRACES_BY_ID: dict[str, AgentTrace] = {t.id: t for t in EXAMPLE_TRACES}
