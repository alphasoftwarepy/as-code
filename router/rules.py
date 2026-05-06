"""
AS Code — Routing Rules & Keyword Maps

Zero-overhead keyword-based routing for model selection.
Designed for real-world latency — no ML inference for routing itself.
"""

# Keywords that strongly indicate reasoning/planning tasks
REASONING_KEYWORDS: frozenset[str] = frozenset({
    # Core reasoning
    "why", "explain", "analyze", "analyse", "debug", "plan",
    "reason", "think", "evaluate", "assess", "consider",
    # Architecture & design
    "architecture", "design", "strategy", "approach", "tradeoff",
    "trade-off", "pattern", "principle", "decision",
    # Investigation
    "investigate", "diagnose", "troubleshoot", "understand",
    "breakdown", "break-down", "compare", "contrast",
    # Review & analysis
    "review", "audit", "critique", "pros", "cons", "advantages",
    "disadvantages", "implications", "consequences",
    # Planning
    "roadmap", "milestone", "phase", "priority", "prioritize",
    "schedule", "timeline", "workflow", "process",
    # Chain-of-thought triggers
    "step-by-step", "reasoning", "logic", "deduce", "infer",
    "hypothesis", "assumption", "conclusion",
})

# Keywords that strongly indicate coding/generation tasks
CODING_KEYWORDS: frozenset[str] = frozenset({
    # Core coding
    "code", "implement", "function", "class", "method", "write",
    "create", "build", "develop", "program", "script",
    # Modification
    "refactor", "optimize", "fix", "patch", "update", "modify",
    "change", "replace", "rename", "move", "delete", "remove",
    # Generation
    "generate", "template", "scaffold", "boilerplate", "snippet",
    "example", "sample", "demo", "prototype",
    # Testing
    "test", "unittest", "pytest", "assert", "mock", "stub",
    "coverage", "spec",
    # Languages & tech
    "python", "javascript", "typescript", "html", "css", "sql",
    "json", "yaml", "toml", "dockerfile", "makefile",
    # Structures
    "api", "endpoint", "route", "handler", "middleware",
    "component", "module", "package", "library", "framework",
    "database", "schema", "migration", "model", "view",
    # Operations
    "install", "deploy", "configure", "setup", "import", "export",
    "parse", "serialize", "validate", "format", "lint",
    # Autocomplete triggers
    "complete", "autocomplete", "suggestion", "hint",
})

# System prompt templates per model role
SYSTEM_PROMPTS: dict[str, str] = {
    "reasoning": (
        "You are a helpful, friendly, general-purpose AI assistant. You can engage in "
        "natural conversation, planning, brainstorming, explanations, and analysis. "
        "Respond naturally and clearly. Only generate code when explicitly requested."
    ),
    "coding": (
        "You are an expert software engineering assistant specialized in programming, "
        "debugging, APIs, architecture, and development workflows. "
        "Write clean, efficient, production-ready code."
    ),
}
