"""
AS Code — Working Memory Manager

Manages short-term agent state: variables, tasks, observations.
All writes are manual (no automatic LLM extraction — infrastructure first).

session_id is the isolation key.  Default: "default_session".
Future: per-chat, per-project, per-VSCode-tab sessions.

format_prompt_block() produces the system-level context block
injected at SYSTEM → Skill → [Working Memory] → RAG → HISTORY → USER.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger("as-code.runtime.memory")

DEFAULT_SESSION = "default_session"


class WorkingMemoryManager:
    """
    Stateless service — every method takes a `db: Session` argument.
    app.state.memory holds this singleton; routes pass their own DB session.
    """

    # ── Variables ──────────────────────────────────────────────

    def set_variable(
        self, db: Session, session_id: str, key: str, value: str
    ) -> dict:
        """Upsert a key-value variable for the session."""
        from api.memory_models import MemoryVariable

        existing = (
            db.query(MemoryVariable)
            .filter_by(session_id=session_id, key=key)
            .first()
        )
        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            logger.debug(f"[MEM] variable updated — session={session_id!r} key={key!r}")
            return existing.to_dict()

        var = MemoryVariable(session_id=session_id, key=key, value=value)
        db.add(var)
        db.commit()
        db.refresh(var)
        logger.debug(f"[MEM] variable created — session={session_id!r} key={key!r}")
        return var.to_dict()

    def delete_variable(self, db: Session, session_id: str, key: str) -> bool:
        from api.memory_models import MemoryVariable

        obj = (
            db.query(MemoryVariable)
            .filter_by(session_id=session_id, key=key)
            .first()
        )
        if not obj:
            return False
        db.delete(obj)
        db.commit()
        logger.debug(f"[MEM] variable deleted — session={session_id!r} key={key!r}")
        return True

    def get_variables(self, db: Session, session_id: str) -> list[dict]:
        from api.memory_models import MemoryVariable

        rows = (
            db.query(MemoryVariable)
            .filter_by(session_id=session_id)
            .order_by(MemoryVariable.key)
            .all()
        )
        return [r.to_dict() for r in rows]

    # ── Tasks ──────────────────────────────────────────────────

    def add_task(
        self,
        db: Session,
        session_id: str,
        title: str,
        priority: int = 0,
    ) -> dict:
        from api.memory_models import MemoryTask

        # next display order
        count = db.query(MemoryTask).filter_by(session_id=session_id).count()
        task = MemoryTask(
            session_id=session_id,
            title=title,
            priority=priority,
            order=count,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        logger.debug(
            f"[MEM] task added — session={session_id!r} "
            f"title={title!r} priority={priority}"
        )
        return task.to_dict()

    def update_task_status(
        self, db: Session, task_id: str, status: str
    ) -> Optional[dict]:
        from api.memory_models import MemoryTask

        valid = {"pending", "in_progress", "completed", "failed"}
        if status not in valid:
            raise ValueError(f"Invalid task status: {status!r}. Must be one of {valid}")

        task = db.query(MemoryTask).filter_by(id=task_id).first()
        if not task:
            return None
        task.status = status
        task.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(task)
        logger.debug(f"[MEM] task status → {status!r} — id={task_id!r}")
        return task.to_dict()

    def delete_task(self, db: Session, task_id: str) -> bool:
        from api.memory_models import MemoryTask

        task = db.query(MemoryTask).filter_by(id=task_id).first()
        if not task:
            return False
        db.delete(task)
        db.commit()
        logger.debug(f"[MEM] task deleted — id={task_id!r}")
        return True

    def get_tasks(self, db: Session, session_id: str) -> list[dict]:
        from api.memory_models import MemoryTask

        rows = (
            db.query(MemoryTask)
            .filter_by(session_id=session_id)
            .order_by(MemoryTask.priority.desc(), MemoryTask.order.asc())
            .all()
        )
        return [r.to_dict() for r in rows]

    # ── Observations ───────────────────────────────────────────

    def add_observation(
        self,
        db: Session,
        session_id: str,
        content: str,
        source: str = "user",
    ) -> dict:
        from api.memory_models import MemoryObservation

        valid_sources = {"user", "system", "rag", "capability"}
        if source not in valid_sources:
            raise ValueError(
                f"Invalid observation source: {source!r}. "
                f"Must be one of {valid_sources}"
            )

        obs = MemoryObservation(
            session_id=session_id, content=content, source=source
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)
        logger.debug(
            f"[MEM] observation added — session={session_id!r} source={source!r}"
        )
        return obs.to_dict()

    def delete_observation(self, db: Session, obs_id: str) -> bool:
        from api.memory_models import MemoryObservation

        obs = db.query(MemoryObservation).filter_by(id=obs_id).first()
        if not obs:
            return False
        db.delete(obs)
        db.commit()
        logger.debug(f"[MEM] observation deleted — id={obs_id!r}")
        return True

    def get_observations(self, db: Session, session_id: str) -> list[dict]:
        from api.memory_models import MemoryObservation

        rows = (
            db.query(MemoryObservation)
            .filter_by(session_id=session_id)
            .order_by(MemoryObservation.created_at.asc())
            .all()
        )
        return [r.to_dict() for r in rows]

    # ── Snapshot ───────────────────────────────────────────────

    def get_memory(self, db: Session, session_id: str) -> dict:
        """Full memory snapshot for a session — used by UI + injection."""
        return {
            "session_id": session_id,
            "variables": self.get_variables(db, session_id),
            "tasks": self.get_tasks(db, session_id),
            "observations": self.get_observations(db, session_id),
        }

    # ── Reset ──────────────────────────────────────────────────

    def reset(self, db: Session, session_id: str) -> None:
        from api.memory_models import MemoryVariable, MemoryTask, MemoryObservation

        db.query(MemoryVariable).filter_by(session_id=session_id).delete()
        db.query(MemoryTask).filter_by(session_id=session_id).delete()
        db.query(MemoryObservation).filter_by(session_id=session_id).delete()
        db.commit()
        logger.info(f"[MEM] session reset — session_id={session_id!r}")

    # ── Prompt Block ───────────────────────────────────────────

    def format_prompt_block(self, db: Session, session_id: str) -> str:
        """
        Produce the Working Memory system prompt block.

        Injection order (defined in api/routes.py):
          SYSTEM PROMPT  =  base_role_prompt
                         +  skill_prompt          (X-Skill header)
                         +  [this block]          ← Working Memory
          MESSAGES       =  chat_history
                         +  [RAG context]         (prepended to last user msg)
                         +  last user message

        Returns an empty string if memory is completely empty (no injection).
        """
        snap = self.get_memory(db, session_id)

        variables = snap["variables"]
        tasks = snap["tasks"]
        observations = snap["observations"]

        # Don't inject anything if memory is empty — no noise
        if not variables and not tasks and not observations:
            return ""

        lines: list[str] = ["## Working Memory\n"]

        if variables:
            lines.append("**Variables:**")
            for v in variables:
                lines.append(f"- {v['key']}: {v['value']}")
            lines.append("")

        if tasks:
            lines.append("**Tasks** (priority high → low):")
            from api.memory_models import MemoryTask
            for t in tasks:
                icon = MemoryTask.STATUS_ICON.get(t["status"], "[ ]")
                prio = f"[P{t['priority']}]" if t["priority"] > 0 else "[P0]"
                lines.append(f"- {icon} {prio} {t['title']}")
            lines.append("")

        if observations:
            lines.append("**Observations:**")
            for o in observations:
                lines.append(f"- [{o['source']}] {o['content']}")
            lines.append("")

        return "\n".join(lines).strip()
