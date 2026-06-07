import logging
from typing import Optional
from runtime.coordinator.models import WorkflowState

logger = logging.getLogger("as-code.runtime.coordinator.workflow_continuation")

RESOLVER_VERSION = "v1_passthrough"


class WorkflowContinuationResolver:
    """
    Architectural extension point responsible for answering:
        Should the active workflow skill be inherited for this turn?

    Separation of concerns:
        intent.py               → What skill does the message activate?
        continuity_resolver.py  → Should retrieval queries be merged?
        workflow_continuation.py → Should the active workflow continue?
        manager.py              → Assemble runtime state

    Phase 1 (v1_passthrough):
        Always returns True. Behavior is identical to current production.
        Emits [WORKFLOW-TRACE] to collect empirical evidence for future
        rule derivation. No continuity rules are applied yet.
    """

    def resolve(
        self,
        user_message: str,
        current_state: WorkflowState,
        inferred_skill: Optional[str],
        manual_skill: Optional[str],
        session_id: str,
    ) -> bool:
        """
        Returns True if the active workflow skill should be inherited this turn.

        In v1_passthrough, always returns True (passthrough behavior).
        Future versions will derive the decision from collected trace evidence.
        """
        decision = True
        reason = "default_passthrough"

        logger.info(
            f"[WORKFLOW-TRACE] "
            f"session_id={session_id} "
            f"message=\"{user_message}\" "
            f"active_skill={current_state.active_skill} "
            f"objective={current_state.objective!r} "
            f"current_phase={current_state.current_phase!r} "
            f"manual_skill={manual_skill} "
            f"inferred_skill={inferred_skill} "
            f"continuation_decision={decision} "
            f"resolver_version={RESOLVER_VERSION} "
            f"reason={reason}"
        )

        return decision
