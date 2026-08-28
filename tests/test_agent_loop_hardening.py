import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from runtime.skills.models import SkillManifest
from runtime.coordinator.models import ContextManifest, RuntimeContract, SessionSnapshot, WorkflowState
from runtime.coordinator.parser import parse_capability_call, KNOWN_CAPABILITY_IDS
from runtime.coordinator.manager import PureCoordinator
from runtime.coordinator.agent import AgentControlRunner
from api.models import ChatCompletionRequest, ChatMessage
from providers.base import InferenceRequest, InferenceResult
from api.streaming import stream_inference_results


def test_skill_manifest_uses_capabilities_default():
    manifest = SkillManifest(
        id="test_skill",
        name="Test Skill",
        description="A test skill"
    )
    assert manifest.uses_capabilities is False


def test_context_manifest_capability_gate_default():
    manifest = ContextManifest(
        contract_id="req-123",
        workflow_state=WorkflowState(),
        rag_enabled=False,
        system_prompt_snapshot=""
    )
    assert manifest.capability_gate_open is False


def test_parser_whitelist_and_schema():
    # Valid call to known capability
    valid_text = '```json_call\n{"capability": "documents", "action": "read", "params": {"document_id": "test.txt"}}\n```'
    call = parse_capability_call(valid_text)
    assert call is not None
    assert call["capability"] == "documents"
    assert call["action"] == "read"

    # Unknown capability rejected by whitelist
    unknown_text = '```json_call\n{"capability": "unregistered_cap", "action": "read", "params": {}}\n```'
    call = parse_capability_call(unknown_text)
    assert call is None

    # Invalid schema missing action
    invalid_schema = '```json_call\n{"capability": "documents"}\n```'
    call = parse_capability_call(invalid_schema)
    assert call is None

    # Raw JSON block with known capability
    raw_text = 'Some text {"capability": "rag", "action": "query", "params": {}} trailing text'
    call = parse_capability_call(raw_text)
    assert call is not None
    assert call["capability"] == "rag"


def test_capability_gate_closed_for_general_chat():
    coord = PureCoordinator()
    db = MagicMock()
    
    # Contract with model_id="chat" (which maps to general)
    contract = RuntimeContract(
        request_id="req-1",
        session_id="sess-1",
        model_id="chat",
        user_message="crea una poesia sobre paraguay de 100 palabras",
        timestamp=123.456,
        snapshot=SessionSnapshot(session_id="sess-1", turn_number=1)
    )

    manifest = coord.assemble(
        db=db,
        contract=contract,
        skill_service=None,
        rag_service=None,
        memory_service=None,
        enable_rag=False
    )

    assert manifest.capability_gate_open is False
    assert "PROTOCOLO DE INVOCACIÓN DE CAPACIDADES" not in manifest.system_prompt_snapshot
    assert "CATÁLOGO DE CAPACIDADES DISPONIBLES" not in manifest.system_prompt_snapshot


def test_capability_gate_opened_for_code_with_skill_uses_capabilities():
    coord = PureCoordinator()
    db = MagicMock()
    
    # Mock skill service returning a skill manifest with uses_capabilities=True
    skill_service = MagicMock()
    skill_manifest = SkillManifest(
        id="programming",
        name="Programming",
        description="Programming skill",
        uses_capabilities=True
    )
    skill_service.get_skill_manifest.return_value = skill_manifest
    skill_service.get_skill_prompt.return_value = "Programming prompt"

    contract = RuntimeContract(
        request_id="req-2",
        session_id="sess-2",
        model_id="code",
        user_message="lee el archivo test.py",
        manual_skill="programming",
        timestamp=123.456,
        snapshot=SessionSnapshot(session_id="sess-2", turn_number=1)
    )

    manifest = coord.assemble(
        db=db,
        contract=contract,
        skill_service=skill_service,
        rag_service=None,
        memory_service=None,
        enable_rag=False
    )

    assert manifest.capability_gate_open is True
    assert "PROTOCOLO DE INVOCACIÓN DE CAPACIDADES" in manifest.system_prompt_snapshot
    assert "CATÁLOGO DE CAPACIDADES DISPONIBLES" in manifest.system_prompt_snapshot


def test_agent_loop_blocks_capability_call_when_gate_closed():
    async def run_test():
        engine = MagicMock()
        # Mock engine returning a JSON capability call
        call_text = '```json_call\n{"capability": "documents", "action": "read", "params": {"document_id": "paraguay_poesia.txt"}}\n```'
        engine.generate = AsyncMock(return_value=InferenceResult(text=call_text, model_id="chat"))
        
        mock_registry = MagicMock()
        runner = AgentControlRunner(engine, registry=mock_registry)
        
        db = MagicMock()
        body = ChatCompletionRequest(
            model="chat",
            messages=[ChatMessage(role="user", content="crea una poesia")]
        )
        inference_request = InferenceRequest(
            prompt="crea una poesia",
            model_id="chat"
        )

        # Run with capability_gate_open=False
        response = await runner.run_inference_loop(
            db=db,
            body=body,
            inference_request=inference_request,
            capability_gate_open=False
        )

        # Capability should NOT be executed
        mock_registry.get.assert_not_called()
        assert response.choices[0].message.content == call_text

    asyncio.run(run_test())


def test_streaming_lifecycle_single_stop_and_done():
    async def run_test():
        async def mock_results():
            yield InferenceResult(text="Hello", model_id="chat")
            yield InferenceResult(text=" world", model_id="chat")
            yield InferenceResult(text="", finish_reason="stop", model_id="chat")

        chunks = []
        async for sse in stream_inference_results(mock_results(), model_id="chat"):
            chunks.append(sse)

        # Verify exactly one [DONE] at the end
        assert chunks[-1] == "data: [DONE]\n\n"
        
        # Verify stop chunk was emitted exactly once
        stop_chunks = [c for c in chunks if '"finish_reason":"stop"' in c or '"finish_reason": "stop"' in c]
        assert len(stop_chunks) == 1

    asyncio.run(run_test())
