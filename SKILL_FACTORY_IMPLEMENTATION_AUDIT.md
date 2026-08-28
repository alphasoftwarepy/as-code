# SKILL FACTORY IMPLEMENTATION AUDIT — AS-CORE

> **Fecha:** 2026-08-28  
> **Revisión:** Gemini 3.7 HIGH  
> **Alcance:** Auditoría arquitectónica preliminar para la creación experimental de Skills temporales y generación de propuestas reproducibles.

---

## 1. Arquitectura Existente

### 1.1 Registro y Carga de Skills Oficiales
- **Ubicación:** Directorio raíz `skills/<skill_id>/`.
- **Estructura por Skill:** Cada skill oficial contiene exactamente:
  - `manifest.json`: Definido por el modelo `SkillManifest` (`id`, `name`, `description`, `required_scopes`, `enabled`, `prompt_family`, `uses_capabilities`).
  - `prompt.md`: Directivas del sistema y modo de operación.
- **Cargador:** `SkillLoader` en `runtime/skills/loader.py` gestiona el singleton global `get_skill_loader()`. Escanea `skills/`, valida manifests y evalúa compatibilidad contra las capacidades dinámicas (`SkillLoader.evaluate_skills()`).

### 1.2 Ejecución de Skills e Inyección de Contexto
- **Coordinador:** `PureCoordinator.assemble()` (`runtime/coordinator/manager.py`) ensambla sin efectos secundarios el `ContextManifest` a partir de un `RuntimeContract`.
- **Inyección:** Si la skill activa está presente, se inyecta su `prompt.md` en el `system_prompt` junto con el anclaje de idioma (`[LANG=...]`), la memoria activa (`WorkingMemoryManager`) y el contexto RAG si aplica.
- **Capability Gate:** Si el modelo es de tipo `agent` o (`coding`/`reasoning`/`moe` con `skill.uses_capabilities=True`), se abre la compuerta de capacidades (`capability_gate_open = True`) y se inyectan el protocolo `json_call` y el catálogo de capacidades activas.

### 1.3 Registro y Ejecución de Capabilities
- **Primitivas:** Heredan de `BaseCapability` (`runtime/capabilities/base.py`) implementando `check()` y `execute()`.
- **Registro:** `CapabilityRegistry` en `runtime/capabilities/registry.py` mantiene las 4 capacidades base: `documents`, `rag`, `git`, `terminal`.
- **Parser:** `runtime/coordinator/parser.py` valida llamadas contra `KNOWN_CAPABILITY_IDS` con esquema JSON estricto (`capability`, `action`, `params`).

### 1.4 Terminal Capability y Gate de Aprobación Humana
- **Terminal:** `TerminalCapability` (`runtime/capabilities/terminal.py`) tiene `scopes = ["terminal.execute"]`, `approval_required_actions = ["execute"]`, y `enabled=False` por defecto a menos que se active explícitamente en `settings.capability_overrides`.
- **Aprobación Humana:** `AgentControlRunner.execute_capability()` (`runtime/coordinator/agent.py`) verifica `cap.requires_approval(action)`. Si requiere aprobación, pausa la ejecución, genera un `approval_id` (`appr-...`) y retorna status `pending_approval` para confirmación interactiva. No existe bypass para acciones protegidas.

### 1.5 Estructura de dev-notes y Tests
- `dev-notes/` contiene ADRs, contratos, arquitectura y bitácoras.
- La suite de pruebas actual (`tests/`) incluye tests de hardening del Agent Loop, contratos, memoria, router y proyectos. Todos los tests de infraestructura y contratos pasan exitosamente (18 passed, 9 skipped sin GPU/GGUF).

---

## 2. Puntos de Extensión

1. **Nuevo módulo `runtime/skills/temporary.py`**:
   - Modelos de datos para el ciclo de vida de una Skill temporal: `TemporarySkillStatus` (`CREATED`, `READY`, `TESTING`, `PASSED`, `FAILED`, `EXPIRED`), `TemporarySkillManifest`, `SkillTestCase`, `SkillTestResult`, `TemporarySkill`.
2. **Nuevo módulo `runtime/skills/factory.py`**:
   - `SkillFactory`: Clase encargada de la instanciación de workspaces temporales, generación de manifests y prompts experimentales, ejecución de pruebas controladas a través del Capability Gateway existente, y renderizado del markdown de propuesta (`ProposalGenerator`).
3. **Workspace de ejecución aislado**:
   - Directorio `temp_skills/<skill_id>/` o `temp/skill_runs/<skill_id>/` conteniendo subdirectorios `workspace/`, `tests/`, `results/`, `logs/`.
4. **Directorio de salida de propuestas**:
   - `dev-notes/skill-proposals/<timestamp>_<skill_id>.md`.
5. **Punto de invocación explícito**:
   - La `SkillFactory` es invocable de forma aislada y bajo demanda (ej. vía servicio o comando explícito). No interfiere con el enrutamiento automático de mensajes comunes en `PureCoordinator`.

---

## 3. Archivos Nuevos y Modificados

### Archivos Nuevos
- `runtime/skills/temporary.py`: Modelos, enums de ciclo de vida y abstracción de Skill temporal.
- `runtime/skills/factory.py`: Lógica de creación, sandboxing, ejecución controlada y generación de proposals.
- `tests/test_skill_factory.py`: Suite completa con los 10 tests unitarios obligatorios y validaciones de invariantes.
- `dev-notes/skill-proposals/`: Directorio para albergar las propuestas generadas para revisión humana.

### Archivos Modificados (Mínimos y Quirúrgicos)
- `runtime/skills/__init__.py`: Exportar clases de conveniencia si es necesario (sin romper imports existentes).
- Ningún archivo de `core/moe/`, `providers/`, `skills/` (oficiales), ni contratos de `InferenceProvider` será modificado.

---

## 4. Estrategia de Aislamiento y Seguridad

1. **Aislamiento de Almacenamiento (No Pollution):**
   - Las skills oficiales residen en `skills/`. `SkillFactory` nunca escribirá en `skills/`, `core/`, o `providers/`.
   - Todas las skills temporales operan estrictamente dentro de su carpeta aislada `temp_skills/<skill_id>/` (o `temp/skill_runs/<skill_id>/`).
   - El `SkillLoader` oficial (`runtime/skills/loader.py`) continúa leyendo exclusivamente de `skills/`, por lo que las skills temporales jamás aparecen en el catálogo oficial de skills sin intervención humana.

2. **Aislamiento de Capacidades y Terminal:**
   - La ejecución de herramientas por parte de una Skill temporal pasa 100% por el `CapabilityRegistry` y el `AgentControlRunner`.
   - No se crean bypasses ni accesos directos a `subprocess` / `os.system` fuera de los canales auditados y sujetos a las políticas de seguridad existentes.

3. **Invariante Human-in-the-Loop:**
   - Toda propuesta generada tendrá `Status: EXPERIMENTAL`, `Human Review Required: YES`, y `Recommendation: PROMOTE_TO_OFFICIAL | NEEDS_REFINEMENT | REJECT`.
   - La promoción a skill oficial sigue siendo una tarea manual realizada a través de Antigravity.

4. **Invariante de Cero Regresión:**
   - Sin habilitación explícita, ninguna consulta de chat, razonamiento o código activará la `SkillFactory`.
