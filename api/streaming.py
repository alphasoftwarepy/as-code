from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

from sqlalchemy.orm import Session

from api.models import ChatCompletionChunk, DeltaContent, StreamChoice
from providers.base import InferenceResult

logger = logging.getLogger("as-code.api.streaming")


def generate_auto_title(user_message: str) -> str:
    """Generates a clean auto-title based on the user's first query if it's long enough."""
    cleaned = user_message.strip().lstrip("¿?¡!#*-_ \t\n\r")
    first_line = cleaned.split("\n")[0].strip()
    words = first_line.split()
    if len(cleaned) < 15 or len(words) < 3:
        return "Nuevo Chat"
    
    # Take first 4-5 words or up to 30 chars
    title_words = []
    char_count = 0
    for w in words:
        if char_count + len(w) > 30 and len(title_words) >= 2:
            break
        title_words.append(w)
        char_count += len(w) + 1
        if len(title_words) >= 5:
            break
            
    title = " ".join(title_words)
    title = title.rstrip(".,;:?¡! ")
    if title:
        title = title[0].upper() + title[1:]
    return title or "Nuevo Chat"


async def stream_inference_results(
    results: AsyncIterator[InferenceResult],
    model_id: str,
    completion_id: str | None = None,
    session_id: str | None = None,
    db: Session | None = None,
) -> AsyncIterator[str]:
    """Convert inference result chunks to SSE-formatted strings.

    Yields strings in the format: "data: {json}\n\n"
    Final message: "data: [DONE]\n\n"

    Guarantees exactly one terminal lifecycle event per generation.
    """
    comp_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    first_chunk = True
    assistant_buffer = []
    terminal_sent = False

    try:
        async for result in results:
            if result.finish_reason == "error":
                # Stream error as a terminal chunk
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
                terminal_sent = True
                break

            if result.text:
                assistant_buffer.append(result.text)
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
                terminal_sent = True
                break

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        if not terminal_sent:
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
            terminal_sent = True

    finally:
        # Guard: if generator ended without stop or error, emit guard stop chunk
        if not terminal_sent:
            guard_chunk = ChatCompletionChunk(
                id=comp_id,
                model=model_id,
                choices=[
                    StreamChoice(
                        delta=DeltaContent(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {guard_chunk.model_dump_json()}\n\n"

        # Persist assistant response and autotitle if database connection is available
        if session_id and db and assistant_buffer:
            full_assistant_text = "".join(assistant_buffer)
            try:
                from runtime.projects.manager import ProjectManager
                pm = ProjectManager()
                pm.add_chat_message(db, session_id, role="assistant", content=full_assistant_text)
                
                # Autotitle checking
                chat = pm.get_chat_by_session(db, session_id)
                if chat and (chat.title == "Nuevo Chat" or chat.title.startswith("Chat ")):
                    msgs = pm.list_chat_messages(db, session_id)
                    user_msgs = [m for m in msgs if m.role == "user"]
                    if user_msgs:
                        first_user_msg = user_msgs[0].content
                        new_title = generate_auto_title(first_user_msg)
                        if new_title and new_title != "Nuevo Chat":
                            pm.rename_chat(db, session_id, new_title)
                            logger.info(f"[AUTOTITLE] Auto-titled chat {session_id} to '{new_title}'")
            except Exception as save_err:
                logger.error(f"Error persisting assistant message or autotitling in stream: {save_err}")

        # Always end with [DONE] exactly once
        yield "data: [DONE]\n\n"

