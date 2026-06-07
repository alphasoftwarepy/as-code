with open('ROADMAP.md', 'r', encoding='utf-8') as f:
    content = f.read()

new_section = (
    "\n---\n\n"
    "## Phase 3.6 \u2014 Intent Routing & Prompt Resolution (Completed)\n\n"
    "A cycle of surgical fixes to the intent routing, persona resolution, and workflow instrumentation layers, "
    "validated through production logs and automated regression tests.\n\n"
    "*   **Session-Scoped RAG (Active Document Context):** RAG retrieval strictly filtered by active `session_id`. "
    "Document uploads preserve the current session. RAG is disabled when no documents exist in the session.\n"
    "*   **User Intent Gate:** `analyze_intent` returns an empty list immediately when the user message contains "
    "no skill-specific keywords. Word-boundary matching (`\\b`) blocks sub-word false positives.\n"
    "*   **Prompt Family Registry:** New `runtime/coordinator/prompts.py` centralizes all prompt templates under "
    "`PROMPT_FAMILIES`. Resolution driven by `prompt_family` in each skill `manifest.json`. No hardcoded skill lists in the coordinator.\n"
    "*   **BUSINESS_PROMPT decoupled from analytical structure:** Removed mandatory DIAGNOSTICO / ANALISIS / ACCION sections. "
    "Now defines domain identity only, letting the LLM adapt output format to the actual task type.\n"
    "*   **WorkflowContinuationResolver (v1_passthrough):** New architectural extension point separating workflow continuity "
    "from retrieval continuity. Emits `[WORKFLOW-TRACE]` logs for evidence collection before continuity rules are derived.\n\n"
    "> See [`dev-notes/STABLE.md`](dev-notes/STABLE.md) for validated invariants that must not be modified without impact analysis.\n\n"
)

marker = "\U0001f6a7 Phase 4"
content = content.replace(marker, new_section + marker, 1)
content = content.replace("(Current Focus)", "(Next)", 1)

with open('ROADMAP.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
