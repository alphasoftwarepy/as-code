import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from api.memory_models import MemoryVariable, MemoryTask
from runtime.coordinator.models import WorkflowState

logger = logging.getLogger("as-code.runtime.coordinator.workflow")

PHASES_BY_SKILL = {
    "marketing": ["discovery", "strategy", "content", "conversion"],
    "business": ["planning", "pricing", "operations"],
    "legal": ["review", "risk", "explanation"],
    "sales": ["pipeline", "pricing", "deal_close"],
    "content_creator": ["scripting", "editing", "publishing"],
    "programming": ["analysis", "implementation", "debugging"]
}

def load_workflow_state(db: Session, session_id: str) -> WorkflowState:
    """Load workflow state variables from Working Memory Variable table."""
    vars_list = db.query(MemoryVariable).filter_by(session_id=session_id).all()
    vars_dict = {v.key: v.value for v in vars_list}

    return WorkflowState(
        objective=vars_dict.get("wf_objective"),
        current_phase=vars_dict.get("wf_phase"),
        current_focus=vars_dict.get("wf_focus"),
        active_skill=vars_dict.get("wf_skill")
    )

def save_workflow_state(db: Session, session_id: str, state: WorkflowState) -> None:
    """Persist workflow state as variables in Working Memory Variable table."""
    from runtime.memory.manager import WorkingMemoryManager
    mem_mgr = WorkingMemoryManager()

    if state.objective is not None:
        mem_mgr.set_variable(db, session_id, "wf_objective", state.objective)
    if state.current_phase is not None:
        mem_mgr.set_variable(db, session_id, "wf_phase", state.current_phase)
    if state.current_focus is not None:
        mem_mgr.set_variable(db, session_id, "wf_focus", state.current_focus)
    if state.active_skill is not None:
        mem_mgr.set_variable(db, session_id, "wf_skill", state.active_skill)

def predict_next_workflow_state(
    user_message: str,
    inferred_skill: Optional[str],
    current_state: WorkflowState
) -> WorkflowState:
    """
    Pure function to predict workflow phase, focus, and objective
    based on user message, inferred skill, and current state.
    NO database writes.
    """
    state = WorkflowState(
        objective=current_state.objective,
        current_phase=current_state.current_phase,
        current_focus=current_state.current_focus,
        active_skill=current_state.active_skill
    )
    msg_lower = user_message.lower()

    # 1. Initialize or swap workflow if a new skill/intent is detected and no active skill exists,
    # or if we are switching skills.
    active_skill = state.active_skill or inferred_skill
    if inferred_skill and inferred_skill != state.active_skill:
        # Clear previous state to prevent contamination when changing skills
        state.objective = None
        state.current_phase = None
        state.current_focus = None
        
        active_skill = inferred_skill
        state.active_skill = active_skill
        # Initialize default phase
        phases = PHASES_BY_SKILL.get(active_skill, ["general"])
        state.current_phase = phases[0]
        # Infer objective from user message if not set
        if not state.objective:
            # Simple heuristic to extract objective
            if len(user_message) < 50:
                state.objective = user_message
            else:
                state.objective = f"Resolve {active_skill} task"
        state.current_focus = f"Initializing {active_skill} workflow"

        logger.info(
            f"[SKILL-TRACE] workflow initializer swapping/initializing: "
            f"prev={current_state.active_skill} -> new={inferred_skill} (phase={phases[0]} focus='{state.current_focus}')"
        )

    if not active_skill:
        return state

    # 2. Heuristic Phase Transitions based on keywords
    phases = PHASES_BY_SKILL.get(active_skill, [])
    current = state.current_phase

    if active_skill == "marketing" and phases:
        # ["discovery", "strategy", "content", "conversion"]
        if current == "discovery" and any(k in msg_lower for k in ["campaña", "estrategia", "plan", "público"]):
            state.current_phase = "strategy"
            state.current_focus = "Defining campaign strategy"
        elif current in ["discovery", "strategy"] and any(k in msg_lower for k in ["flyer", "hook", "crear", "post", "copy", "imagen", "video"]):
            state.current_phase = "content"
            state.current_focus = "Creating hooks and media content"
        elif current == "content" and any(k in msg_lower for k in ["publicar", "conversión", "métrica", "vender", "lanzar"]):
            state.current_phase = "conversion"
            state.current_focus = "Analyzing campaign conversion"

    elif active_skill == "business" and phases:
        # ["planning", "pricing", "operations"]
        if current == "planning" and any(k in msg_lower for k in ["precio", "cobrar", "cuánto", "costar", "pricing"]):
            state.current_phase = "pricing"
            state.current_focus = "Calculating pricing structure"
        elif current in ["planning", "pricing"] and any(k in msg_lower for k in ["operar", "contratar", "procesos", "flujo", "logística"]):
            state.current_phase = "operations"
            state.current_focus = "Setting up business operations"

    elif active_skill == "legal" and phases:
        # ["review", "risk", "explanation"]
        if current == "review" and any(k in msg_lower for k in ["riesgo", "peligro", "responsabilidad", "cláusula"]):
            state.current_phase = "risk"
            state.current_focus = "Analyzing contract risks"
        elif current in ["review", "risk"] and any(k in msg_lower for k in ["explicar", "entender", "qué significa", "resumen"]):
            state.current_phase = "explanation"
            state.current_focus = "Composing contract summary"

    elif active_skill == "sales" and phases:
        # ["pipeline", "pricing", "deal_close"]
        if current == "pipeline" and any(k in msg_lower for k in ["precio", "oferta", "propuesta", "pricing"]):
            state.current_phase = "pricing"
            state.current_focus = "Formulating proposal pricing"
        elif current in ["pipeline", "pricing"] and any(k in msg_lower for k in ["cerrar", "firmar", "close", "ganado"]):
            state.current_phase = "deal_close"
            state.current_focus = "Closing the sales deal"

    elif active_skill == "content_creator" and phases:
        # ["scripting", "editing", "publishing"]
        if current == "scripting" and any(k in msg_lower for k in ["editar", "corte", "audio", "video"]):
            state.current_phase = "editing"
            state.current_focus = "Editing media files"
        elif current in ["scripting", "editing"] and any(k in msg_lower for k in ["publicar", "subir", "postear", "upload"]):
            state.current_phase = "publishing"
            state.current_focus = "Publishing content"

    elif active_skill == "programming" and phases:
        # ["analysis", "implementation", "debugging"]
        if current == "analysis" and any(k in msg_lower for k in ["implementar", "crear", "escribir", "develop", "build"]):
            state.current_phase = "implementation"
            state.current_focus = "Implementing the solution"
        elif current in ["analysis", "implementation"] and any(k in msg_lower for k in ["bug", "error", "falla", "no funciona", "debug", "fix"]):
            state.current_phase = "debugging"
            state.current_focus = "Debugging and fixing issues"

    return state

def update_workflow(
    db: Session,
    session_id: str,
    user_message: str,
    inferred_skill: Optional[str]
) -> WorkflowState:
    """
    Update workflow phase, focus, and objective based on message intent and active skill.
    """
    current_state = load_workflow_state(db, session_id)
    state = predict_next_workflow_state(user_message, inferred_skill, current_state)
    # Save transitions
    save_workflow_state(db, session_id, state)
    return state

def process_task_progression(db: Session, session_id: str, user_message: str) -> None:
    """
    Check if the user is signaling task completion.
    E.g. 'ya tengo los hooks' -> mark task containing 'hooks' as completed.
    And move next pending task to 'in_progress'.
    """
    msg_lower = user_message.lower()

    # Keywords signaling completion
    done_signals = [
        "ya tengo", "listo", "completado", "hecho", "terminado", "ok con", "ok",
        "done", "ready", "finished", "check"
    ]

    is_signaling_done = any(sig in msg_lower for sig in done_signals)
    if not is_signaling_done:
        return

    # Load tasks
    tasks = db.query(MemoryTask).filter_by(session_id=session_id).order_by(MemoryTask.priority.desc(), MemoryTask.order.asc()).all()
    if not tasks:
        return

    completed_any = False
    for task in tasks:
        if task.status in ["completed", "failed"]:
            continue

        # Extract words from task title
        title_words = [w.strip("?,.!-()\"'") for w in task.title.lower().split() if len(w) > 3]
        # Check if user message contains any major word from the task title
        word_match = False
        for tw in title_words:
            if tw in msg_lower:
                word_match = True
                break

        if word_match:
            task.status = "completed"
            logger.info(f"[TASK-PROGRESSION] Task auto-completed: {task.title}")
            completed_any = True
            break  # Only complete one task matching at a time

    if completed_any:
        # Find the next pending task and set it as in_progress
        for next_task in tasks:
            if next_task.status == "pending":
                next_task.status = "in_progress"
                logger.info(f"[TASK-PROGRESSION] Task set to in_progress: {next_task.title}")
                break
        db.commit()
