# General imports
import os
import asyncio
import contextlib
import sys
import re
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Auth imports
import google.auth
import google.auth.transport.requests
from google.auth import compute_engine

# Pipeline imports
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import create_vertex_search_tool, vertex_search
from dotenv import load_dotenv

# Evaluation imports
from langfuse import Langfuse
from aieng.agent_evals.langfuse import init_tracing

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_credentials():
    """Get credentials, preferring the attached GCE service account over ADC.
    
    The workspace's user ADC (from `gcloud auth application-default login`) may
    lack required permissions, so we try the GCE metadata service account first.
    """
    request = google.auth.transport.requests.Request()
    try:
        creds = compute_engine.Credentials(scopes=SCOPES)
        creds.refresh(request)
        console.print("[dim]Auth: using attached GCE service account[/dim]")
        return creds
    except Exception as exc:
        console.print(f"[dim]Auth: GCE service account unavailable; falling back to ADC[/dim]")
        creds, _ = google.auth.default(scopes=SCOPES)
        return creds


async def run_vertex_search_demo(query: str, model: str | None = None) -> dict:
    """Run a demo query against the Vertex AI Search data store."""
    # tool = create_vertex_search_tool(
    #     datastore_id=config.vertex_datastore_id,
    #     location=config.google_cloud_location,
    #     model=model or config.default_worker_model,
    # )
    result = await vertex_search(query)
    return result


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a lowercase set for overlap scoring."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _annotate_summary_with_citations(summary: str, citations: list[dict]) -> tuple[str, list[dict]]:
    """Attach inline numeric citations to summary sentences and return used citations.

    Sentences are matched to citations by keyword overlap with citation snippets.
    """
    if not summary or not citations:
        return summary, []

    sentences = _split_sentences(summary)
    citation_tokens = [_tokenize(c.get("snippet", "")) for c in citations]

    index_by_citation: dict[int, int] = {}
    ordered_used: list[dict] = []
    annotated_sentences: list[str] = []

    for sentence_idx, sentence in enumerate(sentences):
        s_tokens = _tokenize(sentence)
        best_idx = None
        best_score = 0

        for i, c_tokens in enumerate(citation_tokens):
            score = len(s_tokens & c_tokens)
            if score > best_score:
                best_score = score
                best_idx = i

        # Ensure every sentence gets a citation marker.
        if best_idx is None or best_score == 0:
            best_idx = sentence_idx % len(citations)

        if best_idx not in index_by_citation:
            index_by_citation[best_idx] = len(index_by_citation) + 1
            ordered_used.append(citations[best_idx])
        marker = index_by_citation[best_idx]
        annotated_sentences.append(f"{sentence} [{marker}]")

    return " ".join(annotated_sentences), ordered_used


def print_search_results(result: dict) -> None:
    """Display search results in a human-readable format using Rich."""
    status = result.get("status", "unknown")
    
    # Status indicator
    status_icon = "✓" if status == "success" else "✗"
    status_color = "green" if status == "success" else "red"
    console.print(f"[{status_color}]{status_icon}[/{status_color}] Status: {status}")
    
    if status == "error":
        error = result.get("error", "Unknown error")
        console.print(Panel(error, title="Error", border_style="red"))
        return
    
    # Display summary with inline citations
    summary = result.get("summary", "")
    citations = result.get("citations", [])
    annotated_summary, used_citations = _annotate_summary_with_citations(summary, citations)
    if summary:
        console.print(Panel(
            annotated_summary,
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
            expand=True
        ))
    else:
        console.print(Panel(
            "[dim]No summary available[/dim]",
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
            expand=True
        ))
    
    # Display sources
    sources = result.get("sources", [])
    source_count = result.get("source_count", 0)
    
    # Deduplicate sources by URI to avoid showing the same source multiple times
    seen_uris = set()
    unique_sources = []
    for src in sources:
        uri = src.get("uri", "")
        if uri and uri not in seen_uris:
            seen_uris.add(uri)
            unique_sources.append(src)
    
    unique_count = len(unique_sources)
    
    if unique_sources:
        src_table = Table(
            title=f"Unique Sources ({unique_count} unique, {source_count} total citations)",
            show_header=True
        )
        src_table.add_column("#", style="dim", width=3)
        src_table.add_column("Title", style="cyan", width=40)
        src_table.add_column("URI", style="dim", width=50)
        
        for i, src in enumerate(unique_sources, 1):
            title = src.get("title", "")[:38] + "..." if len(src.get("title", "")) > 38 else src.get("title", "")
            uri = src.get("uri", "")
            # Truncate URI for display
            uri_display = uri[:47] + "..." if len(uri) > 47 else uri
            src_table.add_row(str(i), title, uri_display)
        
        console.print(src_table)
    else:
        console.print("[dim]No sources retrieved[/dim]")
    
    # Display metadata
    console.print(f"\n[dim]Response metadata:[/dim]")
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Key", style="dim")
    meta_table.add_column("Value", style="white")
    meta_table.add_row("Unique sources", str(unique_count))
    meta_table.add_row("Total citations", str(source_count))
    meta_table.add_row("Structured citations", str(len(used_citations)))
    console.print(meta_table)

    # Display bibliography in requested format
    if used_citations:
        console.print("\n[bold cyan]Citations[/bold cyan]")
        for i, c in enumerate(used_citations, 1):
            doc_name = c.get("doc_name", "Unknown")
            doc_name_source = c.get("doc_name_source", "unknown source")
            section_title = c.get("section_title", "Unknown")
            section_title_source = c.get("section_title_source", "unknown source")
            page_number = c.get("page_number", "Unknown")
            page_number_source = c.get("page_number_source", "unknown source")
            paragraph_number = c.get("paragraph_number", "Unknown")
            paragraph_number_source = c.get("paragraph_number_source", "unknown source")
            snippet = c.get("snippet", "Unknown")
            console.print(f"[{i}]")
            console.print(f"Doc Name: {doc_name} ({doc_name_source})")
            console.print(f"Section Title: {section_title} ({section_title_source})")
            console.print(f"Page number: {page_number} ({page_number_source})")
            console.print(f"Number of the paragraph: {paragraph_number} ({paragraph_number_source})")
            console.print(f"Short paragraph citation (max 50 words): {snippet}")
            console.print()
    
async def main():
    """Main async function to run the demo."""
    global console, config
    
    if Path("").absolute().name == "eval-agents":
        print(f"Working directory: {Path('').absolute()}")
    else:
        os.chdir(Path("").absolute().parent.parent)
        print(f"Working directory set to: {Path('').absolute()}")

    load_dotenv(verbose=True)
    
    # Initialize credentials early (prefers GCE service account)
    credentials = get_credentials()
    
    config = Configs()  # type: ignore[call-arg]

    if not config.vertex_datastore_id:
        console.print("[red]✗[/red] VERTEX_AI_DATASTORE_ID is not set.")
        console.print("[dim]Set it in your .env file and re-run this cell.[/dim]")
    else:
        cfg_table = Table(title="Vertex AI Search Configuration", show_header=False)
        cfg_table.add_column("Key", style="cyan")
        cfg_table.add_column("Value", style="white")
        cfg_table.add_row("Data store", config.vertex_datastore_id)
        cfg_table.add_row("Region", config.google_cloud_location)
        cfg_table.add_row("Model", config.default_worker_model)
        cfg_table.add_row("Auth", "Application Default Credentials (ADC)")
        console.print(cfg_table)

        # result = await run_vertex_search_demo("What is the inspection frequency for relief valves?")
        # result = await run_vertex_search_demo("What is the minimum required capacity and rating of the portable fire extinguisher each bulk plant must have?")
        result = await run_vertex_search_demo("If a bulk plant stores cylinders awaiting use, resale or exchange in excess of 720 lb, how close must a fire extinguisher be to that storage location?")
        
        # Display results in organized format
        console.print()
        console.print("[bold]Search Results:[/bold]")
        console.print()
        print_search_results(result)


# Initialize console at module level
console = Console(width=100)
config = None

# Run the main async function
if __name__ == "__main__":
    asyncio.run(main())



# Display the result in a more readable format
# console.print(
#     Panel(
#         result["summary"],
#         title="Answer",
#         border_style="cyan",
#     )
# )

# if result["sources"]:
#     src_table = Table(title=f"Sources ({result['source_count']} retrieved)")
#     src_table.add_column("#", style="dim", width=3)
#     src_table.add_column("Title", style="cyan")
#     src_table.add_column("Document", style="dim")

#     for i, src in enumerate(result["sources"], 1):
#         # URI is the full document resource name — the last segment is the document ID
#         doc_id = src["uri"].split("/")[-1] if src["uri"] else ""
#         src_table.add_row(str(i), src["title"], doc_id)

#     console.print(src_table)


# DATASET_NAME = "ionut-is-learning"
# tracing_enabled = init_tracing()
# langfuse = Langfuse()

# qa_items = [
#     {
#         "input": {
#             "question": "What are the pricing tiers?",
#             "user_type": "free",
#             "context": "Product inquiry",
#             "language": "en"
#         },
#         "expected_output": {
#             "answer": "We offer Starter ($29/mo), Professional ($99/mo), and Enterprise (custom pricing).",
#             "confidence": 0.95,
#             "sources": ["pricing_page", "faq"],
#             "follow_up_suggested": False
#         },
#         "metadata": {
#             "difficulty": "easy",
#             "category": "pricing",
#             "domain": "product",
#             "version": "v1.0",
#             "quality_score": 0.9,
#             "priority": "high",
#             "tags": ["pricing", "starter", "faq"],
#             "source_document": "pricing_guide.md",
#             "last_updated": "2024-01-15",
#             "expected_duration_ms": 150
#         },
#         "source_trace_id": "trace-prod-001",
#         "source_observation_id": "obs-pricing-lookup-001",
#         "status": "ACTIVE"
#     },
#     {
#         "input": {
#             "question": "How do I integrate with Salesforce?",
#             "user_type": "enterprise",
#             "context": "Technical integration",
#             "language": "en"
#         },
#         "expected_output": {
#             "answer": "Use our Salesforce connector via AppExchange or REST API. See docs.example.com/salesforce-integration for step-by-step guide.",
#             "confidence": 0.88,
#             "sources": ["integration_guide", "api_docs"],
#             "follow_up_suggested": True
#         },
#         "metadata": {
#             "difficulty": "hard",
#             "category": "technical",
#             "domain": "integrations",
#             "version": "v2.1",
#             "quality_score": 0.85,
#             "priority": "critical",
#             "tags": ["salesforce", "integration", "crm", "enterprise"],
#             "source_document": "salesforce_integration.md",
#             "last_updated": "2024-02-20",
#             "expected_duration_ms": 450
#         },
#         "source_trace_id": "trace-prod-002",
#         "source_observation_id": "obs-salesforce-integration-001",
#         "status": "ACTIVE"
#     },
#     {
#         "input": {
#             "question": "What compliance standards do you meet?",
#             "user_type": "enterprise",
#             "context": "Security & compliance",
#             "language": "en"
#         },
#         "expected_output": {
#             "answer": "SOC 2 Type II, ISO 27001, GDPR, HIPAA, and CCPA compliant.",
#             "confidence": 0.99,
#             "sources": ["compliance_center", "security_whitepaper"],
#             "follow_up_suggested": False
#         },
#         "metadata": {
#             "difficulty": "medium",
#             "category": "compliance",
#             "domain": "security",
#             "version": "v1.5",
#             "quality_score": 0.95,
#             "priority": "critical",
#             "tags": ["compliance", "security", "soc2", "gdpr", "hipaa"],
#             "source_document": "compliance_whitepaper.md",
#             "last_updated": "2024-03-10",
#             "expected_duration_ms": 200
#         },
#         "source_trace_id": "trace-prod-003",
#         "source_observation_id": "obs-compliance-check-001",
#         "status": "ACTIVE"
#     },
# ]

# with contextlib.suppress(Exception):
#     langfuse.create_dataset(name=DATASET_NAME)

# for i, item in enumerate(qa_items):
#     langfuse.create_dataset_item(
#         dataset_name=DATASET_NAME,
#         id=f"{DATASET_NAME}-{i}",
#         input=item["input"],
#         expected_output=item["expected_output"],
#         metadata=item.get("metadata"),
#         source_trace_id=item.get("source_trace_id"),
#         source_observation_id=item.get("source_observation_id"),
#         status=item.get("status", "ACTIVE"),
#     )

# console.print(f"[green]✓[/green] Dataset '[cyan]{DATASET_NAME}[/cyan]' ready ({len(qa_items)} items)")