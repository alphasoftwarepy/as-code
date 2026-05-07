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
        chat_model: str = "chat",
        coding_model: str = "code",
        reasoning_model: str = "reasoning",
        default_model: Optional[str] = None,
    ) -> None:
        self.chat_model = chat_model
        self.coding_model = coding_model
        self.reasoning_model = reasoning_model
        self.default_model = default_model or chat_model

        # Model → role mapping for system prompts
        self._model_roles: dict[str, str] = {
            chat_model: "reasoning",  # 'chat' uses general reasoning prompt
            coding_model: "coding",
            reasoning_model: "reasoning",
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
            role = self._model_roles.get(explicit_model, explicit_model)
            system_prompt = SYSTEM_PROMPTS.get(role, "")
            logger.debug(f"Explicit model: {explicit_model}")
            return explicit_model, system_prompt

        # Priority 2: Keyword-based scoring
        model_id = self._score_message(message)
        role = self._model_roles.get(model_id, model_id)
        system_prompt = SYSTEM_PROMPTS.get(role, "")

        logger.debug(f"Routed to: {model_id} (role={role})")
        return model_id, system_prompt

    def _score_message(self, message: str) -> str:
        """Score message against keyword sets.
        
        Logic:
        - High score in Reasoning keywords -> Reasoning model
        - High score in Coding keywords -> Coding model
        - Tie or low scores -> Chat model (Default)
        """
        # Normalize and tokenize
        words = frozenset(message.lower().split())

        reasoning_score = len(words & REASONING_KEYWORDS)
        coding_score = len(words & CODING_KEYWORDS)

        if reasoning_score > coding_score and reasoning_score > 0:
            return self.reasoning_model
        elif coding_score >= reasoning_score and coding_score > 0:
            return self.coding_model
        else:
            return self.chat_model

    def get_available_models(self) -> list[dict]:
        """Return model metadata for API responses."""
        return [
            {
                "id": self.chat_model,
                "role": "chat",
                "description": "Natural conversation and general information (Gemma Web)",
            },
            {
                "id": self.coding_model,
                "role": "code",
                "description": "Expert coding and technical implementation (Gemma Base)",
            },
            {
                "id": self.reasoning_model,
                "role": "reasoning",
                "description": "Deep analysis and complex reasoning (Gemma Base)",
            },
        ]
