# SKILL FACTORY & TEMPORARY SKILLS — IMPLEMENTATION REPORT

> **Fecha:** 2026-08-28  
> **Estado:** COMPLETADO / EXPERIMENTAL  
> **Revisión:** Gemini 3.7  
> **Human Review Required:** YES  

---

## 1. Arquitectura Implementada

Se implementó una infraestructura experimental, aditiva y estrictamente aislada para la creación, prueba y documentación reproducible de **Skills Temporales** dentro de AS-Core, sin modificar el registry oficial de skills ni alterar los contratos de inferencia y routing existentes.

```
AS-Core (Solicitud Experimental Explícita)
   ↓
SkillFactory (runtime/skills/factory.py)
   ↓
Temporary Skill Sandbox (temp_skills/<skill_id>/)
   ├── manifest.json
   ├── instructions.md
   ├── workspace/
   ├── tests/
   ├── results/
   └── logs/
   ↓
Ejecución Controlada (Test Runner / Sandbox Containment)
   ↓
Capability Gateway & Approval Gate (Sin subprocess bypass)
   ↓
Evaluación de Criterios y Métricas
   ↓
Generación de Propuesta Markdown Reproducible (.md)
   ↓
dev-notes/skill-proposals/<timestamp>_<skill_id>.md
   ↓
HUMAN REVIEW (Antigravity)
```

---

## 2. Archivos Creados y Modificados

### Archivos Nuevos
1. `runtime/skills/temporary.py`:
   - Enums: `TemporarySkillLifecycle` (`CREATED`, `READY`, `TESTING`, `PASSED`, `FAILED`, `EXPIRED`).
   - Modelos Pydantic: `SkillSpec`, `TemporarySkillManifest` (hereda de `SkillManifest`), `SkillTestCase`, `SkillTestResult`, y clase `TemporarySkill`.
2. `runtime/skills/factory.py`:
   - `SkillFactory`: Orquestador de ciclo de vida, sandboxing con verificación estricta de contención anti-traversal (`resolve_sandboxed_path`), ejecución de tests y renderizado de propuestas en Markdown.
   - Excepción de seguridad: `SandboxSecurityError`.
3. `tests/test_skill_factory.py`:
   - Suite de 10 tests unitarios obligatorios que validan creación, estructura, manifests, aislamiento del `SkillLoader`, contención de paths, proposals, manejo de fallos, inmutabilidad de `skills/`, compuerta de terminal y desactivación por defecto.
4. `scratch/run_e2e_skill_factory.py`:
   - Script para la prueba real controlada end-to-end de la skill experimental `csv_data_extractor`.
5. `dev-notes/skill-proposals/`:
   - Directorio de destino para las propuestas de skills experimentales generadas.

### Archivos Modificados
1. `runtime/skills/__init__.py`:
   - Exportación quirúrgica de `SkillFactory`, `TemporarySkill`, `TemporarySkillLifecycle`, `SkillSpec`, `TemporarySkillManifest`, `SkillTestCase`, `SkillTestResult` y `SandboxSecurityError`.

*(Nota: Ningún componente de `core/moe/`, `providers/`, `skills/` oficiales ni `InferenceProvider` fue modificado).*

---

## 3. Estrategia de Aislamiento (Zero Contamination)

- **Aislamiento en Disco:** Las skills temporales se generan exclusivamente bajo `temp_skills/<skill_id>/`.
- **Inmutabilidad de Skills Oficiales:** La carpeta oficial `skills/` permanece 100% inalterada. Los tests confirman que el conjunto de archivos y carpetas oficiales no sufre ninguna modificación.
- **Aislamiento de Discovery:** El `SkillLoader` oficial (`runtime/skills/loader.py`) continúa escaneando únicamente el directorio `skills/`. `SkillLoader.load_skills()` jamás descubre ni indexa skills bajo `temp_skills/`.
- **Path Containment:** El método `SkillFactory.resolve_sandboxed_path()` valida que ninguna ruta pueda escapar del directorio asignado a la skill temporal, bloqueando activamente intentos de path traversal (`..`), rutas absolutas y escapes fuera del sandbox.

---

## 4. Modelo de Seguridad de Capacidades y Terminal

- **Prohibición de Bypass:** Las skills temporales y sus runners de prueba tienen prohibido invocar `subprocess.Popen`, `os.system`, PowerShell o CMD de forma directa.
- **Ruta Única de Ejecución:** Toda acción sobre el sistema host debe solicitarse mediante el `CapabilityRegistry` y despacharse a través de `AgentControlRunner`.
- **Respeto al Approval Contract:** `TerminalCapability` mantiene `approval_required_actions = ["execute"]` y `enabled=False` por defecto. Toda invocación de terminal queda interceptada y retorna `status="pending_approval"` con su respectivo `approval_id`, requiriendo confirmación explícita del operador humano.

---

## 5. Ciclo de Vida de una Skill Temporal

1. **CREATED:** Se valida la `SkillSpec` y se instancia la estructura del sandbox.
2. **READY:** Se escriben `manifest.json` e `instructions.md` dentro de `temp_skills/<skill_id>/`.
3. **TESTING:** Se ejecutan los casos de prueba dentro del workspace aislado.
4. **PASSED / FAILED:** Se evalúan los resultados contra los criterios de aceptación y se determina la recomendación (`APPROVE`, `NEEDS_REFINEMENT` o `REJECT`).
5. **PROPOSAL:** Se emite el archivo `.md` reproducible con metadatos completos y pasos de reproducción manual.
6. **CLEANUP (Opcional):** Se puede purgar el contenido del workspace preservando los registros de auditoría (`results/`, `logs/`, `manifest.json`).

---

## 6. Prueba Real Controlada End-to-End

Se ejecutó de extremo a extremo la creación y validación de una skill temporal real y segura:
- **Skill ID:** `csv_data_extractor`
- **Nombre:** CSV Data Extractor
- **Objetivo:** Recibir un CSV sintético de prueba (`workspace/input.csv`), calcular ingresos totales y extraer lista de productos en formato JSON validado.
- **Resultado:** `PASS` (3 registros procesados, $655.0 total calculado).
- **Métricas:** Tiempo total de ejecución = 1.33 ms.
- **Propuesta Generada:** `dev-notes/skill-proposals/20260828_030217_csv_data_extractor.md`.
- **Verificación de Inmutabilidad:** `skills/` sin cambios; `SkillLoader` libre de contaminación.

---

## 7. Resultados de Tests y Regresión

### Tests Específicos de Skill Factory
Comando: `python -m pytest tests/test_skill_factory.py -v`
- `test_factory_creates_temporary_skill` — **PASSED**
- `test_temporary_skill_has_manifest` — **PASSED**
- `test_temporary_skill_is_not_official` — **PASSED**
- `test_workspace_isolated` — **PASSED**
- `test_proposal_generated` — **PASSED**
- `test_proposal_contains_recipe` — **PASSED**
- `test_failed_skill_generates_failed_proposal` — **PASSED**
- `test_no_official_skill_modification` — **PASSED**
- `test_terminal_respects_capability_gate` — **PASSED**
- `test_factory_disabled_by_default` — **PASSED**

**Total:** 10 passed en 1.08s.

### Suite Completa de Regresión (Non-MoE GGUF)
Comando: `python -m pytest tests/ --ignore-glob="tests/test_moe_*" -v`
- **Resultado:** 28 passed, 9 skipped (pruebas que requieren hardware/servidor externo), 0 failed.
- **Regresión:** **CERO REGRESIÓN (GREEN)**.

---

## 8. Desglose de Estado

### `[IMPLEMENTED]`
- Módulo de modelos y ciclo de vida de Temporary Skills (`runtime/skills/temporary.py`).
- Orquestador `SkillFactory` con sandbox, anti-traversal, runner y generador de proposals (`runtime/skills/factory.py`).
- Integración en `runtime/skills/__init__.py`.
- Suite de 10 tests de aislamiento y seguridad (`tests/test_skill_factory.py`).
- Carpeta `dev-notes/skill-proposals/` y generador de propuestas en Markdown reproducible.
- Validación E2E controlada con `csv_data_extractor`.

### `[MEASURED]`
- Tiempo de ejecución de test y generación de sandbox para `csv_data_extractor`: **1.33 ms**.
- Cobertura de tests del subsystem de factory: **100% de los 10 requisitos obligatorios verificados en verde**.
- Cero archivos modificados en `skills/` tras múltiples ejecuciones de pruebas.

### `[PROJECTED]`
- Futura integración en UI de un panel de revisión de propuestas experimentales.
- Posible soporte para auto-ejecución de pipelines de testing basados en prompts de evaluación con modelos LLM locales cuando el gate esté abierto.

---

## 9. Limitaciones Conocidas y Parada Obligatoria

1. **Sin Auto-Promoción:** Las skills temporales permanecen confinadas como borradores experimentales. Cualquier incorporación a `skills/` requiere revisión y copia manual por parte de un operador humano a través de Antigravity.
2. **Desactivación en Chat Común:** La `SkillFactory` no se dispara automáticamente en conversaciones generales para evitar consumo innecesario de recursos y generación no supervisada.
3. **Parada Obligatoria:** La fase experimental concluye aquí. No se continuará a fases posteriores sin aprobación humana explícita.
