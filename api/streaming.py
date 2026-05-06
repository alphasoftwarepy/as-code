"""
AS Code — SSE Streaming Handler

Server-Sent Events streaming for OpenAI-compatible responses.
Converts provider InferenceResult chunks to SSE-formatted
ChatCompletionChunk JSON payloads.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

from api.models import ChatCompletionChunk, DeltaContent, StreamChoice
from providers.base import InferenceResult

logger = logging.getLogger("as-code.api.streaming")


async def stream_inference_results(
    results: AsyncIterator[InferenceResult],
    model_id: str,
    completion_id: str | None = None,
) -> AsyncIterator[str]:
    """Convert inference result chunks to SSE-formatted strings.

    Yields strings in the format: "data: {json}\n\n"
    Final message: "data: [DONE]\n\n"

    Compatible with OpenAI streaming format expected by
    Cline, Continue, and other VSCode extensions.
    """
    comp_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    first_chunk = True

    try:
        async for result in results:
            if result.finish_reason == "error":
                # Stream error as a special chunk
                error_chunk = ChatCompletionChunk(
                    id=comp_id,
                    model=model_id,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(
                                content=f"[Error: {result.text or 'inference failed'}]"
                            ),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                break

            if result.text:
                chunk = ChatCompletionChunk(
                    id=comp_id,
                    model=model_id,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(
                                role="assistant" if first_chunk else None,
                                content=result.text,
                            ),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                first_chunk = False

            if result.finish_reason == "stop":
                # Final chunk with finish_reason
                final_chunk = ChatCompletionChunk(
                    id=comp_id,
                    model=model_id,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                break

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = ChatCompletionChunk(
            id=comp_id,
            model=model_id,
            choices=[
                StreamChoice(
                    delta=DeltaContent(content=f"\n[Stream error: {e}]"),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    # Always end with [DONE]
    yield "data: [DONE]\n\n"
