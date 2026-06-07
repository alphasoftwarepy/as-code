import asyncio
from typing import AsyncIterator
from runtime.coordinator.parser import parse_capability_call
from runtime.coordinator.agent import AgentControlRunner, MAX_AGENT_STEPS
from api.models import ChatCompletionRequest, ChatMessage
from providers.base import InferenceRequest, InferenceResult
from runtime.capabilities.registry import CapabilityRegistry

# Mock engine to simulate model outputs
class MockEngine:
    def __init__(self, outputs: list):
        self.outputs = outputs
        self.call_count = 0

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return InferenceResult(text=self.outputs[idx], finish_reason="stop")

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResult]:
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        text = self.outputs[idx]
        # Yield in chunks
        chunk_size = 10
        for i in range(0, len(text), chunk_size):
            yield InferenceResult(text=text[i:i+chunk_size])

def test_parser():
    print("Testing parser extraction patterns...")
    # Test valid json_call block
    text1 = """Sure! Here is the git status:\n```json_call\n{\n  "capability": "git",\n  "action": "status",\n  "params": {}\n}\n```"""
    call1 = parse_capability_call(text1)
    assert call1 is not None
    assert call1["capability"] == "git"
    assert call1["action"] == "status"

    # Test raw json fallback
    text2 = """{\n  "capability": "terminal",\n  "action": "execute",\n  "params": {"command": "ls"}\n}"""
    call2 = parse_capability_call(text2)
    assert call2 is not None
    assert call2["capability"] == "terminal"
    assert call2["action"] == "execute"
    assert call2["params"]["command"] == "ls"
    print("Parser tests passed.")

async def test_agent_loop_sync():
    print("Testing synchronous agent loop execution...")
    # Sequence of outputs from model:
    # 1. Ask for git status
    # 2. Complete reasoning based on status
    mock_outputs = [
        'I need to check the status.\n```json_call\n{"capability": "git", "action": "status", "params": {}}\n```',
        'The status shows clean tree. I am done.'
    ]
    engine = MockEngine(mock_outputs)
    registry = CapabilityRegistry() # Registers default caps with placeholders
    runner = AgentControlRunner(engine, registry)
    
    body = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="check git status please")]
    )
    inf_req = InferenceRequest(
        prompt=body.build_prompt(),
        model_id="mock-model",
        stream=False
    )
    
    res = await runner.run_inference_loop(None, body, inf_req)
    assert engine.call_count == 2
    
    # Assert third message in history is the tool output
    # messages[0] = User: check git status please
    # messages[1] = Assistant: I need to check the status...
    # messages[2] = Tool: placeholder output
    assert len(body.messages) >= 3
    tool_msg = body.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.name == "git"
    assert "Placeholder: Git Integration executed action 'status'" in tool_msg.content
    assert "I am done" in res.choices[0].message.content
    print("Sync loop tests passed.")

async def test_agent_loop_max_steps():
    print("Testing MAX_AGENT_STEPS enforcement...")
    # Model keeps outputting git status calls endlessly
    mock_outputs = [
        '```json_call\n{"capability": "git", "action": "status", "params": {}}\n```'
    ] * 5
    engine = MockEngine(mock_outputs)
    registry = CapabilityRegistry()
    runner = AgentControlRunner(engine, registry)
    
    body = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="check git status")]
    )
    inf_req = InferenceRequest(
        prompt=body.build_prompt(),
        model_id="mock-model",
        stream=False
    )
    
    res = await runner.run_inference_loop(None, body, inf_req)
    # Since MAX_AGENT_STEPS is 3, the engine is called exactly 3 times
    assert engine.call_count == MAX_AGENT_STEPS
    print("Max steps tests passed.")

if __name__ == "__main__":
    test_parser()
    asyncio.run(test_agent_loop_sync())
    asyncio.run(test_agent_loop_max_steps())
    print("\nALL TESTS PASSED SUCCESSFULLY!")
