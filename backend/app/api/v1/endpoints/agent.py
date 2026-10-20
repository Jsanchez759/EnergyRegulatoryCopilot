from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy import select

from app.api.dependencies import AgentServiceDep, CurrentUserDep, DbSessionDep
from app.db import SessionFactory
from app.models import Conversation, ConversationMessage
from app.schemas.agent import AgentRequest, ChatMessage
from app.services.errors import ProviderUnavailableError

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/stream",
    summary="Research a question with cited document and web sources",
    response_class=EventSourceResponse,
)
async def research_stream(
    payload: AgentRequest,
    agent_service: AgentServiceDep,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> AsyncIterator[ServerSentEvent]:
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.user_id == current.user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    stored_messages = await db.scalars(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation.id)
        .order_by(ConversationMessage.created_at)
    )
    history = [
        ChatMessage(role=item.role, content=item.content)
        for item in list(stored_messages)[-12:]
        if item.role in {"user", "assistant"}
    ]
    db.add(
        ConversationMessage(conversation_id=conversation.id, role="user", content=payload.question)
    )
    if conversation.title == "Nueva consulta":
        conversation.title = payload.question.strip()[:60]
    conversation.updated_at = datetime.now(UTC)
    await db.commit()

    answer_parts: list[str] = []
    audit = None
    failure = None
    try:
        async for event in agent_service.stream_answer(payload.question, history, current.user.id):
            if event.get("delta"):
                answer_parts.append(str(event["delta"]))
            if event.get("audit"):
                audit = event["audit"]
            yield ServerSentEvent(data=event)
    except ProviderUnavailableError as exc:
        failure = str(exc)
        yield ServerSentEvent(data={"error": failure})
        yield ServerSentEvent(data={"done": True})
    finally:
        content = "".join(answer_parts)
        if failure:
            content = f"No se pudo completar la consulta. {failure}"
        if content:
            async with SessionFactory() as message_session:
                message_session.add(
                    ConversationMessage(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=content,
                        audit=audit,
                    )
                )
                stored_conversation = await message_session.get(Conversation, conversation.id)
                if stored_conversation is not None:
                    stored_conversation.updated_at = datetime.now(UTC)
                await message_session.commit()
