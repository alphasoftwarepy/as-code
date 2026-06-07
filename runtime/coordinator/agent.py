import json
import logging
import uuid
import time
from typing import AsyncIterator, Optional, List
from sqlalchemy.orm import Session

from api.models import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChoice,
    UsageInfo
)
from providers.base import InferenceRequest, InferenceResult
from runtime.capabilities.registry import get_capability_registry
from runtime.coordinator.parser import parse_capability_call
from config.settings import get_settings

logger = logging.getLogger("as-code.runtime.coordinator.agent")

MAX_AGENT_STEPS = 3

class AgentControlRunner:
    def __init__(self, engine, registry=None):
        self.engine = engine
        self.registry = registry or get_capability_registry()

    async def execute_capability(self, capability_id: str, action: str, params: dict, app_state=None) -> dict:
        """Fetch, check, and execute capability using standard envelope."""
        cap = self.registry.get(capability_id)
        if not cap:
            return {
                "success": False,
                "capability": capability_id,
                "action": action,
                "output": f"[Error: Capability '{capability_id}' not found in registry]"
            }

        # Check availability and status dynamically
        settings = get_settings()
        status = cap.check(settings, app_state)
        if not status.available:
            return {
                "success": False,
                "capability": capability_id,
                "action": action,
                "output": f"[Error: Capability '{capability_id}' is offline: {status.reason}]"
            }
        if not status.enabled:
            return {
                "success": False,
                "capability": capability_id,
                "action": action,
                "output": f"[Error: Capability '{capability_id}' is disabled: {status.reason}]"
            }

        try:
            return await cap.execute(action, params)
        except Exception as e:
            logger.error(f"Error executing capability {capability_id}: {e}", exc_info=True)
            return {
                "success": False,
                "capability": capability_id,
                "action": action,
                "output": f"[Error: Execution failed: {str(e)}]"
            }

    def _optimize_context(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Limit max message history length to prevent context bloat (sliding window)."""
        if len(messages) > 12:
            # Preserve system prompt(s) at the beginning
            system_msgs = [m for m in messages if m.role == "system"]
            # Preserve user messages and tool outputs at the end
            last_msgs = messages[-4:]
            logger.info(f"[AGENT-LOOP] Optimizing context: Truncated history from {len(messages)} to {len(system_msgs) + len(last_msgs)} messages")
            return system_msgs + last_msgs
        return messages

    async def run_inference_loop(
        self,
        db: Session,
        body: ChatCompletionRequest,
        inference_request: InferenceRequest,
        app_state=None,
    ) -> ChatCompletionResponse:
        """Run the agent control loop in non-streaming mode."""
        step = 0
        current_request = inference_request
        assistant_content = ""
        
        while step < MAX_AGENT_STEPS:
            step += 1
            logger.info(f"[AGENT-LOOP] Step {step}/{MAX_AGENT_STEPS}")
            
            result = await self.engine.generate(current_request)
            assistant_content += result.text
            
            call = parse_capability_call(result.text)
            if not call:
                # No tool call, generation finished
                break
                
            capability_id = call.get("capability")
            action = call.get("action")
            params = call.get("params", {})
            
            # Execute
            res_envelope = await self.execute_capability(capability_id, action, params, app_state)
            output = res_envelope.get("output", "")
            
            # Record assistant generation and tool execution result in conversation messages
            body.messages.append(ChatMessage(role="assistant", content=result.text))
            body.messages.append(ChatMessage(role="tool", name=capability_id, content=output))
            
            # Optimize context
            body.messages = self._optimize_context(body.messages)
            
            # Build request for next step
            current_request = InferenceRequest(
                prompt=body.build_prompt(),
                model_id=inference_request.model_id,
                temperature=inference_request.temperature,
                max_tokens=inference_request.max_tokens,
                top_p=inference_request.top_p,
                top_k=inference_request.top_k,
                stop_sequences=inference_request.stop_sequences,
                stream=False,
                system_prompt=inference_request.system_prompt,
                request_id=inference_request.request_id,
            )
            
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=inference_request.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=assistant_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )

    async def run_inference_loop_stream(
        self,
        db: Session,
        body: ChatCompletionRequest,
        inference_request: InferenceRequest,
        app_state=None,
    ) -> AsyncIterator[InferenceResult]:
        """Run the agent control loop in streaming mode, yielding tokens."""
        step = 0
        current_request = inference_request
        
        while step < MAX_AGENT_STEPS:
            step += 1
            logger.info(f"[AGENT-LOOP-STREAM] Step {step}/{MAX_AGENT_STEPS}")
            
            assistant_buffer = ""
            async for chunk in self.engine.generate_stream(current_request):
                if chunk.text:
                    assistant_buffer += chunk.text
                    yield chunk
                    
            call = parse_capability_call(assistant_buffer)
            if not call:
                break
                
            capability_id = call.get("capability")
            action = call.get("action")
            params = call.get("params", {})
            
            # Yield visual feedback tokens
            exec_start_token = f"\n\n⚙️ **[Running {capability_id}.{action}...]**\n"
            yield InferenceResult(text=exec_start_token, model_id=inference_request.model_id)
            
            # Execute
            res_envelope = await self.execute_capability(capability_id, action, params, app_state)
            output = res_envelope.get("output", "")
            
            # Yield tool output chunk wrapped in code blocks
            exec_output_token = f"```\n[Output]: {output}\n```\n\n"
            yield InferenceResult(text=exec_output_token, model_id=inference_request.model_id)
            
            # Save messages to conversation history
            body.messages.append(ChatMessage(role="assistant", content=assistant_buffer))
            body.messages.append(ChatMessage(role="tool", name=capability_id, content=output))
            
            # Optimize context
            body.messages = self._optimize_context(body.messages)
            
            # Build next step request
            current_request = InferenceRequest(
                prompt=body.build_prompt(),
                model_id=inference_request.model_id,
                temperature=inference_request.temperature,
                max_tokens=inference_request.max_tokens,
                top_p=inference_request.top_p,
                top_k=inference_request.top_k,
                stop_sequences=inference_request.stop_sequences,
                stream=True,
                system_prompt=inference_request.system_prompt,
                request_id=inference_request.request_id,
            )
            
        # Final stop chunk
        yield InferenceResult(text="", finish_reason="stop", model_id=inference_request.model_id)
