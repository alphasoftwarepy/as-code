"""
AS Code — Graph Trigger

Deterministic trigger to evaluate whether a user query presents a relational
reasoning need that justifies querying the Graph Layer.

Principles:
  - Distinguishes between content queries (RAG) and relational queries (Graph).
  - Purely deterministic: zero LLM calls, zero embeddings, zero external services.
  - Transparent & explainable: returns structured GraphTriggerDecision with exact signals matched.
  - Domain-aware: leverages domain hints (legal, programming, business, research) to contextualize signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from runtime.graph.normalizer import normalize_label


@dataclass
class GraphTriggerDecision:
    """Structured decision explaining whether and why Graph should be queried."""
    needed: bool
    reason: str
    signals: List[str] = field(default_factory=list)
    confidence: float = 1.0


class GraphTrigger:
    """
    Evaluates whether a query requires relational reasoning.
    
    Relational categories evaluated:
      1. Explicit relational terminology ('relacionado con', 'vinculado a', 'conectado').
      2. Connection questions ('qué conecta X con Y', 'cómo se relaciona X con Y').
      3. Multi-hop and path questions ('a través de', 'cadena de', 'camino entre', 'indirectamente').
      4. Dependency & hierarchical links ('depende de', 'módulos que dependen', 'subordinado a').
      5. Cross-entity attribution ('quién representa a X en Y', 'firmante de X en Y').
    """

    # Pure content patterns that belong strictly to RAG (negative filters)
    RAG_CONTENT_PATTERNS = [
        r'^\s*¿?\s*(?:qué|que)\s+dice\s+(?:este|el|la)?\s*(?:contrato|documento|texto|archivo|cláusula|sección)',
        r'^\s*¿?\s*(?:resume|resumir|resumen)\s+(?:este|el|la|los|las|del)?\s*(?:documento|contrato|texto|archivo)s?',
        r'^\s*¿?\s*(?:qué|que)\s+establece\s+la\s+cláusula',
        r'^\s*¿?\s*(?:cómo|como)\s+funciona\s+(?:este|el|esta|la)?\s*(?:código|función|método|script)',
        r'^\s*¿?\s*explí?came\s+(?:este|el|esta|la)?\s*(?:código|texto|documento|función)',
        r'^\s*¿?\s*(?:qué|que)\s+personas?\s+aparecen?\s+en\s+el\s+contrato',
        r'^\s*¿?\s*(?:qué|que)\s+empresas?\s+aparecen?\s+en\s+(?:estos?|el)?\s*documentos?',
    ]

    # Explicit relational patterns (Spanish & English)
    RELATIONAL_PATTERNS = [
        # Direct relationship terms
        (r'\b(?:relaci[oó]n|relacionad[oa]s?|relaciona|vinculad[oa]s?|v[íi]nculos?|conectad[oa]s?|conexi[oó]n|conexiones)\b', "explicit_relation"),
        (r'\b(?:asociad[oa]s?|asocia|asociaci[oó]n|afiliad[oa]s?)\b', "association_link"),
        (r'\b(?:depende|dependencia|dependencias|dependiente|depends|dependency)\b', "dependency_link"),
        
        # Connection patterns: X with Y, between X and Y
        (r'\b(?:c[oó]mo\s+se\s+relaciona|qu[ée]\s+conecta|qu[ée]\s+vincula|how\s+is\s+.*related)\b', "connection_question"),
        (r'\b(?:entre\s+[A-ZÁÉÍÓÚa-záéíóú\s]+\s+y\s+[A-ZÁÉÍÓÚa-záéíóú\s]+)\b', "between_entities"),
        (r'\b(?:between\s+.*\s+and\s+.*)\b', "between_entities_en"),
        
        # Multi-hop and indirect chains
        (r'\b(?:a\s+trav[ée]s\s+de|por\s+medio\s+de|cadena\s+de\s+relaciones|relaci[oó]n\s+indirecta|indirectamente|through|indirectly)\b', "multi_hop_chain"),
        (r'\b(?:camino\s+entre|recorrido\s+entre|path\s+between)\b', "path_query"),
        
        # Attribution / representation / multi-entity role queries
        (r'\b(?:qui[ée]n\s+representa\s+a\b.*\ben\b|who\s+represents)\b', "representation_query"),
        (r'\b(?:qu[ée]\s+documentos?\s+conectan?|qu[ée]\s+documentos?\s+vinculan?)\b', "cross_document_connection"),
        (r'\b(?:en\s+com[uú]n\s+entre|compartid[oa]s?\s+entre|in\s+common\s+between)\b', "shared_entities"),
    ]

    # Domain-specific relational cues
    DOMAIN_PATTERNS = {
        "programming": [
            (r'\b(?:importa|imported\s+by|llama\s+a|invoca|calls|hereda\s+de|inherits|subclase)\b', "code_dependency"),
            (r'\b(?:qu[ée]\s+m[oó]dulo\s+depende|qu[ée]\s+clase\s+depende)\b', "module_dependency"),
        ],
        "legal": [
            (r'\b(?:parte\s+firmante\s+de|firm(?:aron|ó|o)\s+ambos|aparece\s+en\s+m[aá]s\s+de\s+un\s+contrato)\b', "legal_multi_contract"),
            (r'\b(?:qu[ée]\s+contratos?\s+est[aá]n?\s+relacionados?)\b', "contract_relation"),
        ],
        "business": [
            (r'\b(?:proveedor\s+de\b.*\bcliente|vende\s+a\b.*\ba\s+trav[ée]s\s+de)\b', "business_supply_chain"),
            (r'\b(?:qu[ée]\s+clientes?\s+est[aá]n?\s+conectados?)\b', "customer_connection"),
        ],
        "research": [
            (r'\b(?:co-autores?|colaboraci[oó]n\s+entre|citad[oa]\s+por)\b', "research_collaboration"),
        ]
    }

    def evaluate(
        self,
        query: str,
        domain: Optional[str] = None,
        document_count: Optional[int] = None,
    ) -> GraphTriggerDecision:
        """
        Evaluate user query deterministically.
        Returns GraphTriggerDecision with boolean needed, reason, and list of matched signals.
        """
        if not query or not query.strip():
            return GraphTriggerDecision(needed=False, reason="Empty query", signals=[])

        query_clean = query.strip()
        query_lower = query_clean.lower()

        # 1. Check for pure content / RAG-only patterns (False-Positive Prevention)
        for pattern in self.RAG_CONTENT_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Ensure it does not also have strong explicit multi-hop signals
                if not any(w in query_lower for w in ["a través de", "cadena", "relaciona con", "conectan a"]):
                    return GraphTriggerDecision(
                        needed=False,
                        reason="Content/Summary query naturally served by RAG",
                        signals=["rag_content_query"],
                        confidence=0.95,
                    )

        matched_signals: List[str] = []

        # 2. Check general relational patterns
        for pattern, signal_name in self.RELATIONAL_PATTERNS:
            if re.search(pattern, query_clean, re.IGNORECASE):
                matched_signals.append(signal_name)

        # 3. Check domain-specific relational cues if domain provided
        if domain and domain.lower() in self.DOMAIN_PATTERNS:
            for pattern, signal_name in self.DOMAIN_PATTERNS[domain.lower()]:
                if re.search(pattern, query_clean, re.IGNORECASE):
                    matched_signals.append(signal_name)

        # 4. Contextual signal weighting
        # If document_count > 1 and multiple entities appear linked by 'y' or 'con'
        if document_count is not None and document_count >= 2:
            if re.search(r'\b(?:en\s+ambos|en\s+todos|entre\s+los\s+documentos)\b', query_lower):
                matched_signals.append("cross_document_scope")

        # Decision
        if matched_signals:
            # Format clear human-readable reason
            signals_str = ", ".join(matched_signals)
            return GraphTriggerDecision(
                needed=True,
                reason=f"Relational query detected ({signals_str})",
                signals=matched_signals,
                confidence=1.0,
            )

        return GraphTriggerDecision(
            needed=False,
            reason="No relational requirement identified in query",
            signals=[],
            confidence=1.0,
        )
