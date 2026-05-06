"""
AS Code — Smart Router

Intent-based model routing with zero ML overhead.
Routes requests to the optimal model based on keyword analysis.

Design decisions:
- Keyword matching, NOT embedding similarity (zero latency overhead)
- Frozenset lookups for O(1) per-word checks
- Configurable default model for ambiguous requests
- Explicit model override via request parameter
"""

from __future__ import annotations

import logging
from typing import Optional

from router.rules import (
    CODING_KEYWORDS,
    REASONING_KEYWORDS,
    SYSTEM_PROMPTS,
)

logger = logging.getLogger("as-code.router")


class SmartRouter:
    """Routes inference requests to the optimal model.

    Routing logic (in priority order):
    1. Explicit model specified in request → use that model
    2. Keyword scoring → highest-scoring category wins
    3. Tie or ambiguous → default model (coding)
    """

    def __init__(
        self,
        reasoning_model: str = "deepseek-r1-1.5b",
        coding_model: str = "gemma-4-e2b",
        default_model: Optional[str] = None,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.coding_model = coding_model
        self.default_model = default_model or coding_model

        # Model → role mapping for system prompts
        self._model_roles: dict[str, str] = {
            reasoning_model: "reasoning",
            coding_model: "coding",
        }

    def route(
        self,
        message: str,
        explicit_model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Route a message to the optimal model.

        Args:
            message: The user's message text.
            explicit_model: If set, bypasses routing logic.

        Returns:
            Tuple of (model_id, system_prompt).
        """
        # Priority 1: Explicit model override
        if explicit_model and explicit_model != "auto":
            role = self._model_roles.get(explicit_model, "coding")
            system_prompt = SYSTEM_PROMPTS.get(role, "")
            logger.debug(f"Explicit model: {explicit_model}")
            return explicit_model, system_prompt

        # Priority 2: Keyword-based scoring
        model_id = self._score_message(message)
        role = self._model_roles.get(model_id, "coding")
        system_prompt = SYSTEM_PROMPTS.get(role, "")

        logger.debug(f"Routed to: {model_id} (role={role})")
        return model_id, system_prompt

    def _score_message(self, message: str) -> str:
        """Score message against keyword sets.

        Uses word-level matching with frozenset intersection
        for O(min(n, m)) performance where n = words, m = keywords.
        """
        # Normalize and tokenize
        words = frozenset(message.lower().split())

        reasoning_score = len(words & REASONING_KEYWORDS)
        coding_score = len(words & CODING_KEYWORDS)

        if reasoning_score > coding_score:
            return self.reasoning_model
        elif coding_score > reasoning_score:
            return self.coding_model
        else:
            # Tie: default to coding model (more common use case)
            return self.default_model

    def get_available_models(self) -> list[dict]:
        """Return model metadata for API responses."""
        return [
            {
                "id": self.reasoning_model,
                "role": "reasoning",
                "description": "Reasoning, planning, debugging, analysis",
            },
            {
                "id": self.coding_model,
                "role": "coding",
                "description": "Code generation, implementation, refactoring",
            },
        ]
