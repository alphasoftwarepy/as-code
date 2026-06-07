import logging
from sqlalchemy.orm import Session
from runtime.coordinator.models import RuntimeContract, ContextManifest
from runtime.coordinator.workflow import save_workflow_state, process_task_progression
from api.memory_models import MemoryVariable, MemoryTask, MemoryObservation

logger = logging.getLogger("as-code.runtime.coordinator.mutator")

class RuntimeStateMutator:
    @staticmethod
    def apply_state_mutations(db: Session, contract: RuntimeContract, manifest: ContextManifest) -> None:
        """
        Applies all state mutations in a single post-inference atomic block.
        Includes memory limit enforcement, task progression, workflow updates, and suggestions saving.
        """
        session_id = contract.session_id
        user_message = contract.user_message
        
        try:
            logger.info(f"[MUTATOR] Starting post-inference state mutations for session={session_id} request={contract.request_id}")
            
            if contract.explicit_reset:
                from runtime.memory.manager import WorkingMemoryManager
                WorkingMemoryManager().reset(db, session_id)
                logger.info(f"[MUTATOR] Explicit reset applied — cleared session={session_id}")
            
            # 1. Enforce Memory Limits
            RuntimeStateMutator.enforce_memory_limits(db, session_id)
            
            # 2. Process Task Progression based on user message content
            process_task_progression(db, session_id, user_message)
            
            # 3. Save predicted workflow state
            save_workflow_state(db, session_id, manifest.workflow_state)
            
            # 4. Save suggestions
            from runtime.memory.manager import WorkingMemoryManager
            mem_mgr = WorkingMemoryManager()
            mem_mgr.set_variable(db, session_id, "wf_suggestions", ",".join(manifest.suggested_skills))
            
            # 5. Save continuity decision if present
            if manifest.continuity_decision:
                from runtime.coordinator.state_store import LightweightStateStore
                state_store = LightweightStateStore(db)
                state_store.persist_decision(
                    session_id=session_id,
                    turn_number=contract.snapshot.turn_number if contract.snapshot else 1,
                    final_rag_query=manifest.continuity_decision.final_rag_query,
                    language=manifest.continuity_decision.detected_language,
                    decision=manifest.continuity_decision,
                    request_id=contract.request_id
                )
            
            db.commit()
            logger.info(f"[MUTATOR] Successfully completed post-inference state mutations for session={session_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"[MUTATOR] Failed applying state mutations: {e}", exc_info=True)

    @staticmethod
    def enforce_memory_limits(db: Session, session_id: str) -> None:
        """Enforce strict memory limits to avoid DB bloat & token pollution."""
        # 1. Variables: max 15 (excluding workflow variables wf_*)
        variables = db.query(MemoryVariable).filter_by(session_id=session_id).order_by(MemoryVariable.created_at.asc()).all()
        user_vars = [v for v in variables if not v.key.startswith("wf_")]
        if len(user_vars) > 15:
            excess_count = len(user_vars) - 15
            for i in range(excess_count):
                logger.info(f"[LIMIT-ENFORCE] Trimming old variable: {user_vars[i].key}")
                db.delete(user_vars[i])

        # 2. Tasks: max 10
        tasks = db.query(MemoryTask).filter_by(session_id=session_id).order_by(MemoryTask.created_at.asc()).all()
        if len(tasks) > 10:
            excess_count = len(tasks) - 10
            # Try to delete completed/failed tasks first
            completed_tasks = [t for t in tasks if t.status in ["completed", "failed"]]
            pending_tasks = [t for t in tasks if t.status not in ["completed", "failed"]]
            
            deleted_count = 0
            for ct in completed_tasks:
                if deleted_count < excess_count:
                    db.delete(ct)
                    deleted_count += 1
            
            if deleted_count < excess_count:
                # Delete oldest pending tasks if still over limit
                for pt in pending_tasks:
                    if deleted_count < excess_count:
                        db.delete(pt)
                        deleted_count += 1

        # 3. Observations: max 20
        observations = db.query(MemoryObservation).filter_by(session_id=session_id).order_by(MemoryObservation.created_at.asc()).all()
        if len(observations) > 20:
            excess_count = len(observations) - 20
            for i in range(excess_count):
                db.delete(observations[i])
