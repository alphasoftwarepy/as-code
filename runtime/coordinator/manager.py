import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from api.memory_models import MemoryVariable, MemoryTask, MemoryObservation
from runtime.coordinator.models import WorkflowState, CoordinatorDecision, RuntimeContract, ContextManifest
from runtime.coordinator.intent import analyze_intent
from runtime.coordinator.workflow import load_workflow_state, update_workflow, process_task_progression, predict_next_workflow_state
from runtime.coordinator.suggestions import get_suggested_skills
from runtime.coordinator.continuity_resolver import DeterministicContinuityResolver
from runtime.coordinator.prompts import resolve_root_prompt
from runtime.coordinator.workflow_continuation import WorkflowContinuationResolver


logger = logging.getLogger("as-code.runtime.coordinator")

class RuntimeCoordinator:
    def __init__(self):
        pass

    def coordinate(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        manual_skill: Optional[str] = None
    ) -> CoordinatorDecision:
        """
        Main coordination loop:
        1. Enforce hard memory limits to prevent bloat.
        2. Auto-progress tasks based on message content.
        3. Analyze user message intent.
        4. Resolve active skill using priority order: manual -> persistent workflow -> inferred.
        5. Update workflow state (transitions phases/focus).
        6. Get suggestions and persist them in variables.
        7. Format system prompt runtime context.
        """
        # 1. Enforce limits first
        self.enforce_memory_limits(db, session_id)

        # 2. Process task progression
        process_task_progression(db, session_id, user_message)

        # 3. Analyze intent for skills
        inferred_skills = analyze_intent(user_message, db, session_id)
        first_inferred = inferred_skills[0] if inferred_skills else None

        # 4. Resolve active skill
        # Priority order:
        #   (a) Manually activated skill (e.g. from header X-Skill)
        #   (b) Persistent workflow skill in Working Memory (wf_skill)
        #   (c) Top inferred skill from user message intent
        current_state = load_workflow_state(db, session_id)
        
        resolved_skill = manual_skill
        if not resolved_skill:
            if first_inferred:
                resolved_skill = first_inferred
            elif current_state.active_skill:
                wf_resolver = WorkflowContinuationResolver()
                if wf_resolver.resolve(
                    user_message=user_message,
                    current_state=current_state,
                    inferred_skill=first_inferred,
                    manual_skill=manual_skill,
                    session_id=session_id,
                ):
                    resolved_skill = current_state.active_skill

        # 5. Update workflow state transitions
        workflow_state = update_workflow(db, session_id, user_message, resolved_skill)

        # 6. Get suggestions for alternative skills and persist them in memory
        suggested = get_suggested_skills(db, session_id, user_message, workflow_state)
        
        from runtime.memory.manager import WorkingMemoryManager
        mem_mgr = WorkingMemoryManager()
        mem_mgr.set_variable(db, session_id, "wf_suggestions", ",".join(suggested))

        # 7. Generate runtime context block
        runtime_context = self.build_runtime_context_block(workflow_state, resolved_skill)

        return CoordinatorDecision(
            resolved_skill=resolved_skill,
            suggested_skills=suggested,
            workflow_state=workflow_state,
            runtime_context=runtime_context
        )

    def enforce_memory_limits(self, db: Session, session_id: str) -> None:
        """Enforce strict memory limits to avoid DB bloat & token pollution."""
        try:
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

            db.commit()
        except Exception as e:
            logger.warning(f"Error enforcing memory limits: {e}")

    def build_runtime_context_block(self, state: WorkflowState, active_skill: Optional[str]) -> str:
        """Build the runtime context block to inject into the system prompt."""
        lines = ["## RUNTIME CONTEXT"]

        # Only inject objective if it was explicitly declared by the user (not a generic heuristic)
        if state.objective:
            is_generic = (
                state.objective == f"Resolve {active_skill} task"
                or state.objective == f"Resolve {state.active_skill} task"
                or "workflow" in state.objective.lower()
                or state.objective == active_skill
                or state.objective == state.active_skill
            )
            if not is_generic:
                lines.append(f"Active objective: {state.objective}")

        # current_phase / current_focus are excluded by default to avoid prompt pollution

        if active_skill:
            lines.append(f"Active skill: {active_skill}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines).strip()

class PureCoordinator:
    def __init__(self):
        pass

    def assemble(
        self,
        db: Session,
        contract: RuntimeContract,
        skill_service = None,
        rag_service = None,
        memory_service = None,
        enable_rag: bool = True
    ) -> ContextManifest:
        """
        Stateless & side-effect free assembly of the system prompt.
        """
        # 1. Analyze user intent (read-only)
        inferred_skills = analyze_intent(contract.user_message, db, contract.session_id)
        first_inferred = inferred_skills[0] if inferred_skills else None

        # 2. Resolve skill and workflow state
        current_state = load_workflow_state(db, contract.session_id)
        resolved_skill = contract.manual_skill
        if not resolved_skill:
            if first_inferred:
                resolved_skill = first_inferred
            elif current_state.active_skill:
                wf_resolver = WorkflowContinuationResolver()
                if wf_resolver.resolve(
                    user_message=contract.user_message,
                    current_state=current_state,
                    inferred_skill=first_inferred,
                    manual_skill=contract.manual_skill,
                    session_id=contract.session_id,
                ):
                    resolved_skill = current_state.active_skill

        logger.info(
            f"[SKILL-TRACE] skill resolver: "
            f"manual={contract.manual_skill}, "
            f"workflow_active={current_state.active_skill}, "
            f"inferred={first_inferred} "
            f"-> resolved={resolved_skill}"
        )

        # 3. Predict next workflow state (pure function)
        predicted_wf = predict_next_workflow_state(contract.user_message, resolved_skill, current_state)

        # 4. Get suggestions
        suggested = get_suggested_skills(db, contract.session_id, contract.user_message, predicted_wf)

        # 5. Continuity & Language Resolution
        resolver = DeterministicContinuityResolver()
        decision = resolver.resolve(contract)
        lang = decision.detected_language
        rag_query = decision.final_rag_query

        # 6. Root Prompt (via Prompt Family resolution)
        prompt_family = None
        if resolved_skill and skill_service:
            manifest = skill_service.get_skill_manifest(resolved_skill)
            if manifest:
                prompt_family = manifest.prompt_family

        if contract.model_id == "code" or resolved_skill == "programming":
            prompt_family = "SOFTWARE_PROMPT"

        root_prompt = resolve_root_prompt(lang, prompt_family)

        system_prompt = f"[LANG={lang}]\n{root_prompt}"

        # 7. Inject Skill Prompt
        if resolved_skill and skill_service:
            skill_prompt = skill_service.get_skill_prompt(resolved_skill)
            if skill_prompt:
                system_prompt = f"{system_prompt}\n\n{skill_prompt}"

        # 7.5 Inject Capability Instructions (Cognitive Prompt for Phase 3.5)
        if lang == "ES":
            capability_instructions = (
                "\n\n### PROTOCOLO DE INVOCACIÓN DE CAPACIDADES\n"
                "Si necesitas usar una capacidad del sistema local, genera un bloque JSON envuelto en triple acento grave y el identificador 'json_call'. Detén la generación de inmediato tras cerrar el bloque.\n"
                "Formato:\n"
                "```json_call\n"
                "{\n"
                "  \"capability\": \"<capability_id>\",\n"
                "  \"action\": \"<action_name>\",\n"
                "  \"params\": {}\n"
                "}\n"
                "```\n"
            )
        else:
            capability_instructions = (
                "\n\n### CAPABILITY INVOCATION PROTOCOL\n"
                "If you need to use a local system capability, emit a JSON block wrapped in triple backticks and the 'json_call' identifier. Stop generation immediately after closing the block.\n"
                "Format:\n"
                "```json_call\n"
                "{\n"
                "  \"capability\": \"<capability_id>\",\n"
                "  \"action\": \"<action_name>\",\n"
                "  \"params\": {}\n"
                "}\n"
                "```\n"
            )
        system_prompt = f"{system_prompt}{capability_instructions}"

        # 8. Inject Coordinator context
        runtime_context = self.build_runtime_context_block(predicted_wf, resolved_skill)
        if runtime_context:
            if lang == "ES":
                runtime_context = runtime_context.replace("Active objective:", "Objetivo activo:")
                runtime_context = runtime_context.replace("Current phase:", "Fase actual:")
                runtime_context = runtime_context.replace("Current focus:", "Enfoque actual:")
                runtime_context = runtime_context.replace("Active skill:", "Habilidad activa:")
                runtime_context = runtime_context.replace("pipeline", "embudo de ventas")
                runtime_context = runtime_context.replace("Resolve sales task", "Resolver tarea de ventas")
                runtime_context = runtime_context.replace("conversion", "conversión")
            system_prompt = f"{system_prompt}\n\n{runtime_context}"

        # 9. Inject Working Memory
        memory_block = ""
        memory_vars_cnt = 0
        memory_tasks_cnt = 0
        memory_obs_cnt = 0
        if memory_service:
            memory_block = memory_service.format_prompt_block(db, contract.session_id)
            if memory_block:
                system_prompt = f"{system_prompt}\n\n{memory_block}"
            
            from api.memory_models import MemoryVariable, MemoryTask, MemoryObservation
            try:
                memory_vars_cnt = db.query(MemoryVariable).filter_by(session_id=contract.session_id).count()
                memory_tasks_cnt = db.query(MemoryTask).filter_by(session_id=contract.session_id).count()
                memory_obs_cnt = db.query(MemoryObservation).filter_by(session_id=contract.session_id).count()
            except Exception:
                pass

        # 10. Inject RAG Context
        if rag_service and enable_rag:
            mode = "thinking" if contract.model_id == "reasoning" else ("code" if contract.model_id == "code" else "normal")
            pipeline = "code" if contract.model_id == "code" else "chat"
            try:
                context = rag_service.build_context(
                    query=rag_query,
                    db=db,
                    mode=mode,
                    pipeline=pipeline,
                    session_id=contract.session_id,
                )
                if context:
                    system_prompt = f"{system_prompt}\n\n{context}"
            except Exception:
                pass

        # 11. Localize Headers
        if lang == "ES":
            if "## RUNTIME CONTEXT" in system_prompt:
                system_prompt = system_prompt.replace("## RUNTIME CONTEXT", "## CONTEXTO")
            if "## Working Memory" in system_prompt:
                system_prompt = system_prompt.replace("## Working Memory", "## MEMORIA ACTIVA")
            if "## CONTEXT FROM DOCUMENTS" in system_prompt:
                system_prompt = system_prompt.replace("## CONTEXT FROM DOCUMENTS", "## DOCUMENTOS")
            elif "## RESEARCH CONTEXT" in system_prompt:
                system_prompt = system_prompt.replace("## RESEARCH CONTEXT", "## DOCUMENTOS")

        # 12. Limit checking
        char_budget = 16000
        char_count = len(system_prompt)
        if char_count > char_budget:
            system_prompt = system_prompt[:char_budget]
            char_count = len(system_prompt)

        return ContextManifest(
            contract_id=contract.request_id,
            active_skill=resolved_skill,
            workflow_state=predicted_wf,
            suggested_skills=suggested,
            rag_enabled=enable_rag and (rag_service is not None),
            rag_query=rag_query if enable_rag else None,
            rag_hits=[],
            memory_variables_count=memory_vars_cnt,
            memory_tasks_count=memory_tasks_cnt,
            memory_observations_count=memory_obs_cnt,
            char_budget=char_budget,
            char_count=char_count,
            system_prompt_snapshot=system_prompt,
            continuity_decision=decision
        )

    def build_runtime_context_block(self, state: WorkflowState, active_skill: Optional[str]) -> str:
        lines = ["## RUNTIME CONTEXT"]

        # Only inject objective if it was explicitly declared by the user (not a generic heuristic)
        if state.objective:
            is_generic = (
                state.objective == f"Resolve {active_skill} task"
                or state.objective == f"Resolve {state.active_skill} task"
                or "workflow" in state.objective.lower()
                or state.objective == active_skill
                or state.objective == state.active_skill
            )
            if not is_generic:
                lines.append(f"Active objective: {state.objective}")

        # current_phase / current_focus are excluded by default to avoid prompt pollution

        if active_skill:
            lines.append(f"Active skill: {active_skill}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines).strip()

