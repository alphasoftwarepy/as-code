import re
from typing import List, Dict
from sqlalchemy.orm import Session
from api.memory_models import MemoryObservation
from api.rag_models import RAGDocument

KEYWORD_MAPS: Dict[str, List[str]] = {
    "marketing": [
        "marketing", "ventas", "vender", "instagram", "flyer", "anuncio", "campaña", "campaign",
        "leads", "ad", "target", "audience", "redes", "social media", "publicidad", "hooks",
        "hook", "branding", "copys", "audiencia", "facebook", "twitter", "linkedin", "tiktok", "viral"
    ],
    "sales": [
        "vender", "ventas", "comprar", "sales", "negociar", "pipeline", "deals", "crm", "pricing",
        "precios", "clientes", "prospect", "funnel", "conversión", "oferta", "propuesta", "pitch"
    ],
    "legal": [
        "contrato", "legal", "ley", "cláusula", "ndia", "contract", "acuerdo", "firma", "riesgo",
        "términos", "condiciones", "política", "abogado", "demanda", "propiedad intelectual",
        "regulaciones", "normativa", "clause", "nda", "compliance"
    ],
    "business": [
        "business", "negocio", "estrategia", "planning", "planificación", "operaciones", "operations",
        "presupuesto", "budget", "startup", "empresa", "sociedad", "inversor", "finanzas", "funding",
        "roi", "ganancia", "ingresos", "revenue", "costos"
    ],
    "content_creator": [
        "video", "guión", "script", "post", "blog", "content", "creador", "diseño", "flyer", "imagen",
        "youtube", "tiktok", "copywriting", "redactar", "escribir", "podcast", "thumbnail", "contenido"
    ],
    "programming": [
        "código", "code", "programar", "programación", "bug", "error", "debug", "debuggear",
        "función", "función", "function", "clase", "class", "módulo", "module", "api",
        "implementar", "implement", "refactor", "arquitectura", "architecture", "algoritmo",
        "algorithm", "python", "javascript", "typescript", "sql", "html", "css", "json",
        "endpoint", "backend", "frontend", "test", "unittest", "git", "deploy", "runtime",
        "desarrollar", "desarrollo", "desarrollador", "coding", "scripting", "script", "scripts",
        "base de datos", "database", "server", "fastapi", "compilar", "compiler", "pyproject.toml",
        "requirements.txt", "package.json", "npm", "pip", "docker", "github", "repo", "repository"
    ]
}

def analyze_intent(user_message: str, db: Session, session_id: str) -> List[str]:
    """
    Heuristically analyze user message and workspace state to match skill IDs.
    Returns matching skill IDs ordered by relevance score descending.
    """
    scores: Dict[str, int] = {skill: 0 for skill in KEYWORD_MAPS}
    msg_lower = user_message.lower()

    # Track skills with explicit keyword matches in the user message (User Intent Gate)
    message_matched_skills = set()

    # 1. Match against user message
    for skill, keywords in KEYWORD_MAPS.items():
        for kw in keywords:
            # Word boundary search to avoid sub-word matching issues
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = len(re.findall(pattern, msg_lower))
            if matches > 0:
                scores[skill] += matches * 2
                message_matched_skills.add(skill)

    # User Intent Gate: If no skill keyword matches the user message, it is a neutral/general query.
    # Return immediately to avoid specialized skill pollution from documents/observations.
    if not message_matched_skills:
        import logging
        logging.getLogger("as-code.runtime.coordinator.intent").info(
            f"[SKILL-TRACE] analyze_intent (neutral query gated): msg='{user_message}' -> matched_skills=[]"
        )
        return []

    # 2. Match against active RAG documents (contextual boost, only for message-matched skills)
    try:
        # Enforce Active Retrieval Scope (Phase 3.6) by filtering by session_id
        docs = db.query(RAGDocument).filter_by(session_id=session_id).all()
        for doc in docs:
            filename_lower = doc.filename.lower()
            # Clean filename by replacing hyphens, underscores, dots, etc. with space for boundary search
            filename_clean = re.sub(r'[^a-zA-Z0-9áéíóúüñ]', ' ', filename_lower)
            for skill in message_matched_skills:
                for kw in KEYWORD_MAPS[skill]:
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, filename_clean):
                        scores[skill] += 3
    except Exception:
        # Graceful degradation if RAG table is missing or errors
        pass

    # 3. Match against recent observations in Working Memory (contextual boost, only for message-matched skills)
    try:
        observations = db.query(MemoryObservation).filter_by(session_id=session_id).all()
        for obs in observations:
            obs_lower = obs.content.lower()
            for skill in message_matched_skills:
                for kw in KEYWORD_MAPS[skill]:
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, obs_lower):
                        scores[skill] += 1
    except Exception:
        pass

    # Return matching skills with score > 0, sorted descending
    matched = [skill for skill in message_matched_skills if scores[skill] > 0]
    matched.sort(key=lambda s: scores[s], reverse=True)

    import logging
    logging.getLogger("as-code.runtime.coordinator.intent").info(
        f"[SKILL-TRACE] analyze_intent: msg='{user_message}' -> matched_skills={matched} (scores={scores})"
    )

    return matched
