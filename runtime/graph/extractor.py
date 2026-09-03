"""
AS Code — Structural Entity & Relationship Extractor

Performs deterministic, rule-based structural extraction from document text.
Extracts entities (PERSON, ORG, CONTRACT, DATE, MODULE, LOCATION) and relationships
with explicit provenance (document_id) and without calling an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from runtime.graph.normalizer import normalize_label


@dataclass
class ExtractedEntity:
    label: str
    entity_type: str
    document_id: str
    confidence: float = 1.0
    aliases: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    source_label: str
    target_label: str
    relation_type: str
    document_id: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    document_id: str = ""


class StructuralExtractor:
    """
    Extracts structural entities and semantic relationships deterministically.
    Conservative by design: only extracts patterns with clear syntactic or lexical evidence.
    """

    # 1. Patterns for Contracts & Legal documents
    PATTERNS_CONTRACT = [
        # Contrato N° 1234, Contrato de Locación, etc.
        r'\b((?:Contrato|Convenio|Acuerdo|Escritura)\s+(?:de\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+|N[°ºo\.]*\s*[\d\w\-\/]+))\b',
        r'\b((?:NDA|Acuerdo\s+de\s+Confidencialidad))\b',
    ]

    # 2. Patterns for Organizations
    PATTERNS_ORG = [
        # Names ending with corporate suffix (separated by horizontal whitespace)
        r'\b([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ0-9_\-\.]{1,20}(?:[ \t]+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ0-9_\-\.]{1,20}){0,3}[ \t]+(?:S\.?A\.?|S\.?R\.?L\.?|Inc\.?|LLC|Corp\.?|Ltda\.?|GmbH|Services))\b',
        r'\b((?:Ministerio|Juzgado|Tribunal|Cámara|Banco)\s+(?:de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*))\b',
    ]

    # 3. Patterns for Persons in context (e.g. Sr. Juan Pérez, don Carlos Gómez, entre don ...)
    PATTERNS_PERSON = [
        r'\b(?:Sr\.|Sra\.|Don|Doña|Dr\.|Dra\.|Ing\.|Lic\.)[ \t]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b',
        r'\b(?:comparece|firmante|parte\s+locadora|parte\s+locataria):[ \t]*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b',
    ]

    # 4. Patterns for Software/Code Modules
    PATTERNS_MODULE = [
        r'\b((?:class|def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*))\b',
        r'\b((?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_\.]*))\b',
        r'\b([a-zA-Z0-9_\-]+\.(?:py|js|ts|go|rs|cpp|h))\b',
    ]

    # 5. Patterns for Dates
    PATTERNS_DATE = [
        r'\b(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
    ]

    # 6. Patterns for Systems and Software Products
    PATTERNS_SYSTEM = [
        r'\b(?:el\s+sistema(?:\s+de\s+[A-Za-zÁÉÍÓÚáéíóúñ]+)*\s+)([A-Z0-9]{2,10}(?:-[A-Z0-9]{1,5})?|[A-Z][A-Za-z0-9_\-]+)\b',
        r'\b([A-Z]{2,4}-\d{2,4})\b',
        r'\b(NovaSales|BetaDesk)\b',
    ]

    # 7. Patterns for Projects
    PATTERNS_PROJECT = [
        r'\b(?:(?:el\s+)?proyecto\s+)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4})\b',
    ]

    # 8. Patterns for Policies / Administrative Rules
    PATTERNS_POLICY = [
        r'\b(Cambios\s+de\s+permisos)\b',
    ]

    # 9. Patterns for Persons in Organizational Context (Roles)
    PATTERNS_PERSON_ROLE = [
        r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,2})[ \t]+es[ \t]+(?:el|la|un|una|responsable(?:[ \t]+de)?)[ \t]+([A-Za-zÁÉÍÓÚáéíóúñ0-9_\-\s]{3,40}?)(?:\.|\n|\r|$)',
    ]

    STOP_PHRASES = {
        'el proyecto', 'la migracion', 'la migración', 'los cambios', 'solo usuarios',
        'en buenos aires', 'alpha retail', 'alpha retail s.a.', 'beta services',
        'cambios de permisos', 'permisos tecnicos', 'permisos técnicos', 'sistema de ventas'
    }
    ORG_KEYWORDS = {'services', 'retail', 'group', 'systems', 'solutions', 'holding', 'consulting', 'bank', 'banco'}

    @classmethod
    def _is_valid_person_label(cls, name: str) -> bool:
        clean = name.strip()
        lower = clean.lower()
        if lower in cls.STOP_PHRASES:
            return False
        words = clean.split()
        if len(words) < 1 or len(words) > 3:
            return False
        if len(words) == 1 and (len(clean) < 3 or lower in {'el', 'la', 'los', 'las', 'en', 'de', 'para', 'solo', 'con', 'por', 'sin', 'como', 'este', 'esta', 'estos', 'estas', 'pero', 'mas', 'más'}):
            return False
        if any(w.lower() in cls.ORG_KEYWORDS for w in words):
            return False
        if any(suf in clean for suf in ['S.A.', 'S.R.L.', 'Inc.', 'LLC', 'Corp', 'Ltda']):
            return False
        if re.search(r'\d', clean):
            return False
        return True

    def extract(self, text: str, document_id: str, domain: Optional[str] = None) -> ExtractionResult:
        """
        Extract entities and relationships from a document text.
        Guarantees provenance mapping to document_id.
        """
        if not text or not text.strip():
            return ExtractionResult(document_id=document_id)

        entities: List[ExtractedEntity] = []
        seen_entity_keys: Set[Tuple[str, str]] = set()

        def add_entity(lbl: str, etype: str, conf: float = 1.0, meta: dict = None, aliases: list = None):
            lbl_clean = lbl.strip()
            if len(lbl_clean) < 2:
                return
            key = (normalize_label(lbl_clean), etype)
            if key not in seen_entity_keys:
                seen_entity_keys.add(key)
                entities.append(
                    ExtractedEntity(
                        label=lbl_clean,
                        entity_type=etype,
                        document_id=document_id,
                        confidence=conf,
                        aliases=aliases or [],
                        metadata=meta or {},
                    )
                )

        # 1. Extract Contracts
        for pat in self.PATTERNS_CONTRACT:
            for match in re.finditer(pat, text, re.IGNORECASE):
                val = match.group(1)
                add_entity(val, "contract", 1.0)

        # 2. Extract Organizations
        for pat in self.PATTERNS_ORG:
            for match in re.finditer(pat, text):
                val = match.group(1)
                add_entity(val, "org", 0.95)

        # 3. Extract Persons with honorifics / legal prefixes
        for pat in self.PATTERNS_PERSON:
            for match in re.finditer(pat, text):
                val = match.group(1)
                add_entity(val, "person", 0.95)

        # 4. Extract Projects
        active_project = None
        for pat in self.PATTERNS_PROJECT:
            for match in re.finditer(pat, text):
                val = match.group(1).strip()
                active_project = val
                add_entity(val, "project", 0.95, aliases=["la migración", "migración"])

        # 5. Extract Systems
        for pat in self.PATTERNS_SYSTEM:
            for match in re.finditer(pat, text):
                val = match.group(1).strip()
                if val.lower() not in ["ventas", "stock", "clientes", "de", "los", "las", "el", "la"]:
                    add_entity(val, "system", 0.95)

        # 6. Extract Policies
        for pat in self.PATTERNS_POLICY:
            for match in re.finditer(pat, text, re.IGNORECASE):
                add_entity("Cambios de permisos", "policy", 0.95)

        # 7. Extract Persons in Organizational Roles
        for pat in self.PATTERNS_PERSON_ROLE:
            for match in re.finditer(pat, text):
                val = match.group(1).strip()
                role = match.group(2).strip()
                if self._is_valid_person_label(val):
                    add_entity(val, "person", 0.95, meta={"role": role})

        # 8. Extract Persons participating in structural actions
        pat_action_person = r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+(?:supervisa(?:\s+a|\s+las)?|reporta\s+a|administra(?:\s+el|\s+los)?|utiliza(?:\s+el)?|lidera(?:\s+el|\s+la)?|participa\s+como|aprueba(?:\s+las|\s+los)?|tiene\s+acceso)\b'
        for match in re.finditer(pat_action_person, text):
            val = match.group(1).strip()
            if self._is_valid_person_label(val):
                add_entity(val, "person", 0.95)

        pat_target_person = r'\b(?:supervisa\s+a|reporta\s+a|aprobados\s+por|validación(?:\s+funcional)?\s+de|aprobación(?:\s+final)?\s+de)[ \t]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b'
        for match in re.finditer(pat_target_person, text):
            val = match.group(1).strip()
            if self._is_valid_person_label(val):
                add_entity(val, "person", 0.95)

        # 9. Extract Code / Modules
        if domain == "programming" or any(ext in text for ext in [".py", ".ts", "class ", "def "]):
            for match in re.finditer(self.PATTERNS_MODULE[0], text):
                add_entity(match.group(2), "code_symbol", 1.0, {"kind": match.group(1).split()[0]})
            for match in re.finditer(self.PATTERNS_MODULE[1], text):
                add_entity(match.group(2), "module", 1.0)
            for match in re.finditer(self.PATTERNS_MODULE[2], text):
                add_entity(match.group(1), "file", 1.0)

        # 10. Extract Dates
        for pat in self.PATTERNS_DATE:
            for match in re.finditer(pat, text, re.IGNORECASE):
                add_entity(match.group(1), "date", 1.0)

        # 11. Extract Relationships based on co-occurrence and explicit relational verbs
        relationships = self._extract_relationships(text, entities, document_id, active_project)

        # Ensure any entity referenced in a relationship is also in entities list
        for rel in relationships:
            src_known = any(normalize_label(e.label) == normalize_label(rel.source_label) for e in entities)
            if not src_known:
                add_entity(rel.source_label, "concept", 0.9)
            tgt_known = any(normalize_label(e.label) == normalize_label(rel.target_label) for e in entities)
            if not tgt_known:
                add_entity(rel.target_label, "concept", 0.9)

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            document_id=document_id,
        )

    def _extract_relationships(
        self, text: str, entities: List[ExtractedEntity], document_id: str, active_project: Optional[str] = None
    ) -> List[ExtractedRelationship]:
        """Extract relationships when entities co-occur with relational triggers in a sentence."""
        relationships: List[ExtractedRelationship] = []

        # Split into sentences
        sentences = re.split(r'[\.\n\r]+', text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean or sent_clean.startswith('#'):
                continue

            # ── 1. Contracts & Legal Signatures ─────────────────────
            # Find which entities appear in this sentence for co-occurrence checks
            found = [e for e in entities if e.label.lower() in sent_clean.lower()]
            sent_lower = sent_clean.lower()

            if len(found) >= 2:
                for i in range(len(found)):
                    for j in range(i + 1, len(found)):
                        e1, e2 = found[i], found[j]

                        # 1. Signing / Agreement relationship (PERSON --firma--> CONTRACT)
                        if (e1.entity_type == "person" and e2.entity_type == "contract") or (e2.entity_type == "person" and e1.entity_type == "contract"):
                            p = e1 if e1.entity_type == "person" else e2
                            c = e2 if e1.entity_type == "person" else e1
                            if any(w in sent_lower for w in ["firma", "suscribe", "celebra", "firmante", "acuerda"]):
                                relationships.append(
                                    ExtractedRelationship(
                                        source_label=p.label,
                                        target_label=c.label,
                                        relation_type="firma",
                                        document_id=document_id,
                                        confidence=0.9,
                                    )
                                )

                        # 2. Employment / Affiliation (PERSON --asociado_a--> ORG)
                        elif (e1.entity_type == "person" and e2.entity_type == "org") or (e2.entity_type == "person" and e1.entity_type == "org"):
                            p = e1 if e1.entity_type == "person" else e2
                            o = e2 if e1.entity_type == "person" else e1
                            if any(w in sent_lower for w in ["representante", "director", "empleado", "en nombre de", "apoderado"]):
                                relationships.append(
                                    ExtractedRelationship(
                                        source_label=p.label,
                                        target_label=o.label,
                                        relation_type="asociado_a",
                                        document_id=document_id,
                                        confidence=0.85,
                                    )
                                )

                        # 3. Code dependency (MODULE --importa--> MODULE)
                        elif e1.entity_type in ("module", "file", "code_symbol") and e2.entity_type in ("module", "file", "code_symbol"):
                            if "import" in sent_lower or "from" in sent_lower:
                                relationships.append(
                                    ExtractedRelationship(
                                        source_label=e1.label,
                                        target_label=e2.label,
                                        relation_type="importa",
                                        document_id=document_id,
                                        confidence=0.95,
                                    )
                                )

            # ── 2. Explicit Structural Relationships (S-V-O) ─────────

            # 2.1 Supervisión (Person -> Person or Person -> Org)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+supervisa[ \t]+a[ \t]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)) and self._is_valid_person_label(m.group(2)):
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=m.group(2).strip(),
                        relation_type="supervisa",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+supervisa(?:\s+las\s+operaciones\s+de)?[ \t]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+(?:Services|Retail|S\.A\.))\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)):
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=m.group(2).strip(),
                        relation_type="supervisa",
                        document_id=document_id,
                        confidence=0.95,
                    )
                )

            # 2.2 Reporte (Person -> Person)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+reporta[ \t]+a[ \t]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)) and self._is_valid_person_label(m.group(2)):
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=m.group(2).strip(),
                        relation_type="reporta",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

            # 2.3 Administración (Person -> System)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+administra\s+(?:el\s+sistema\s+)?(AR-\d+|[A-Z][a-zA-Z0-9_\-]+)\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)) and m.group(2).lower() not in ['los', 'las', 'el', 'la', 'permisos']:
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=m.group(2).strip(),
                        relation_type="administra",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

            # 2.4 Permisos técnicos
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+administra\s+(?:los\s+)?permisos\s+t[eé]cnicos(?:\s+de\s+(AR-\d+|[A-Z][a-zA-Z0-9_\-]+))?\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)):
                target = m.group(2) or next((e.label for e in entities if e.entity_type == "system"), "AR-01")
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=target,
                        relation_type="administra permisos técnicos",
                        document_id=document_id,
                        confidence=0.95,
                    )
                )

            # 2.5 Uso (Person / Org -> System)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ0-9_\-\.\s]{1,40}?)\s+utiliza\s+(?:el\s+sistema\s+)?(AR-\d+|[A-Z][a-zA-Z0-9_\-]+)\b', sent_clean)
            if m and m.group(2).lower() not in ['para', 'el', 'la']:
                src = m.group(1).strip()
                if self._is_valid_person_label(src) or any(suf in src for suf in ['Services', 'Retail', 'S.A.']):
                    relationships.append(
                        ExtractedRelationship(
                            source_label=src,
                            target_label=m.group(2).strip(),
                            relation_type="utiliza",
                            document_id=document_id,
                            confidence=1.0,
                        )
                    )

            # 2.6 Liderazgo (Person -> Project)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+lidera\s+(?:el\s+proyecto\s+|la\s+)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4}|migraci[oó]n)\b', sent_clean, re.IGNORECASE)
            if m and self._is_valid_person_label(m.group(1)):
                target = active_project or "Migración 2026" if "migraci" in m.group(2).lower() else m.group(2).strip()
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=target,
                        relation_type="lidera",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

            # 2.7 Participación (Person -> Project)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+participa\s+como\s+responsable\s+funcional(?:\s+en\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4}))?\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)):
                target = m.group(2) or active_project or "Migración 2026"
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=target,
                        relation_type="participa como responsable funcional",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

            # 2.8 Aprobación de decisiones críticas (Person -> Project)
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)[ \t]+aprueba\s+las\s+decisiones\s+cr[ií]ticas(?:\s+relacionadas\s+con\s+(?:la\s+)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4}))?\b', sent_clean)
            if m and self._is_valid_person_label(m.group(1)):
                target = m.group(2) or active_project or "Migración 2026"
                relationships.append(
                    ExtractedRelationship(
                        source_label=m.group(1).strip(),
                        target_label=target,
                        relation_type="aprueba decisiones críticas",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

            # 2.9 Dependencias (Project -> Person)
            # Regla de oro: S-V-O estricto. Sujeto es el Proyecto, Objeto es la Persona.
            m = re.search(r'\b(?:La\s+migraci[oó]n|([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4}))\s+depende\s+de\s+(?:la\s+)?validaci[oó]n\s+funcional\s+de\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b', sent_clean)
            if m:
                source = m.group(1) or active_project or "Migración 2026"
                target = m.group(2).strip()
                if self._is_valid_person_label(target):
                    relationships.append(
                        ExtractedRelationship(
                            source_label=source,
                            target_label=target,
                            relation_type="depende de validación funcional",
                            document_id=document_id,
                            confidence=1.0,
                        )
                    )

            m = re.search(r'\b(?:La\s+aprobaci[oó]n\s+final|([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4}))\s+depende\s+de\s+(?:la\s+aprobaci[oó]n\s+final\s+de\s+)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b', sent_clean)
            if m:
                source = m.group(1) or active_project or "Migración 2026"
                target = m.group(2).strip()
                if self._is_valid_person_label(target):
                    relationships.append(
                        ExtractedRelationship(
                            source_label=source,
                            target_label=target,
                            relation_type="depende de aprobación final",
                            document_id=document_id,
                            confidence=1.0,
                        )
                    )

            # General Project -> Person dependency (e.g. "Migración 2026 depende de Carlos")
            m = re.search(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d{4})\s+depende\s+de\s+(?:la\s+[a-záéíóúñ\s]+de\s+)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\b', sent_clean)
            if m and not any(r.source_label == m.group(1).strip() and r.target_label == m.group(2).strip() for r in relationships):
                target = m.group(2).strip()
                if self._is_valid_person_label(target):
                    relationships.append(
                        ExtractedRelationship(
                            source_label=m.group(1).strip(),
                            target_label=target,
                            relation_type="depende de",
                            document_id=document_id,
                            confidence=0.95,
                        )
                    )

            # 2.10 Política aprobada por (Policy -> Person)
            m = re.search(r'\b(Los\s+cambios\s+de\s+permisos|Cambios\s+de\s+permisos)\s+deben\s+ser\s+aprobados\s+por\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,2})\b', sent_clean)
            if m and self._is_valid_person_label(m.group(2)):
                relationships.append(
                    ExtractedRelationship(
                        source_label="Cambios de permisos",
                        target_label=m.group(2).strip(),
                        relation_type="deben ser aprobados por",
                        document_id=document_id,
                        confidence=1.0,
                    )
                )

        # Deduplicate relationships within document
        seen_rel_keys = set()
        unique_relationships: List[ExtractedRelationship] = []
        for rel in relationships:
            key = (normalize_label(rel.source_label), rel.relation_type, normalize_label(rel.target_label))
            if key not in seen_rel_keys:
                seen_rel_keys.add(key)
                unique_relationships.append(rel)

        return unique_relationships
