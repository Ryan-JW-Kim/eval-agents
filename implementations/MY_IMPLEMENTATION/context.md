Goal 
The project involves implementing two major feature groups: Preprocessing and agentic QA.

First, we have to take the PDF document and preprocess it. This means either producing high quality OCR of the pages, or if it is a text PDF decoding the document to be markdown or html; call this our “intermediate representation”. After the document is in the intermediate representation, we shall chunk it using semantic chunking, after which an LLM (coding assistant or small LLM workflow) will be used to generate more structured claims for a given chunk. Very important we preserve the metadata traceability for each preprocessing step we take. The original PDF has discrete pages, the resulting intermediate representation shall share the page-by-page breakdown, and then each chunk shall be a member of one or multiple pages, and each LLM generated claim shall be a member of a given chunk and will reference exact text verbatim as its intermediate representation “source content”.

Second, we will implement a set of tools for an agent to use during a question + answer session/run. A single “run” shall have one “user query” and one “final response”. During the agent execution it will have multiple iterations and different tools to answer the user query to the best of its ability. Tools such as thinking, query, and final_response. “Thinking” is a generic agent helper tool, since modern models have hidden reasoning traces this might be redundant but shall be provided for the agent to be able to explicitly inject self-instructions into the conversation memory. “Query” shall be implemented asynchronously and will invoke either a RAG or GraphRAG (design decision needed, raise to developer!). “Final_response” is the tool call used when the agent wants to end the iteration because it has the answer ready, this tool shall have a field for raw markdown text that will be used to validate against the ground truth.

Coding standards
The project will be written using python, all blob data shall be stored locally in JSON/JSONL files, all configuration data shall be stored using yaml.
Project structure
.
├── config.yaml
├── agent/
│   ├── ... (rip out Vector Institute reference implementation)
│   └── prompts/
│       ├── system.md
│       ├── query_tool.md
│       ├── thinking_tool.md
│       └── final_response_tool.md
├── evals/
│   ├── readme.md
│   ├── ground_truths.jsonl
│   ├── heuristic/
│   │   ├── h1_safety_matches.py
│   │   ├── h2_response_semantic_diff.py
│   │   └── h3_citation_accuracy.py
│   ├── llm/
│   │   ├── l1_response_score.py
│   │   ├── l2_response_relevance.py
│   │   └── l3_trace_rating.py
│   └── framework/
│       └── ... infrastructure to facilitate evals on agent.
├── data/
│   ├── unstructured/
│   │   ├── original_document.pdf
│   │   └── intermediate_representation.md
│   └── structured/
│       ├── repr_chunked.jsonl
│       ├── all_text_units.jsonl
│       └── train_text_units.jsonl (provided to the query tool)
└── logs/
    └── ...each run shall have bundled logs named by UUID + timestamp/
        ├── config.yaml (copy of the config used)
        ├── prompts (copy of the prompts used)/
        │   └── ...
        ├── trace.jsonl
        ├── user_query.md
        └── final_response.md 
