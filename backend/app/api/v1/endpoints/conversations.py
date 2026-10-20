from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import or_, select

from app.api.dependencies import CurrentUserDep, DbSessionDep
from app.models import Conversation, ConversationMessage
from app.schemas.agent import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationMessagesResponse,
    ConversationSummary,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    db: DbSessionDep,
    current: CurrentUserDep,
    search: Annotated[str | None, Query(max_length=100)] = None,
) -> ConversationListResponse:
    statement = select(Conversation).where(Conversation.user_id == current.user.id)
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        matching_messages = select(ConversationMessage.conversation_id).where(
            ConversationMessage.content.ilike(pattern)
        )
        statement = statement.where(
            or_(Conversation.title.ilike(pattern), Conversation.id.in_(matching_messages))
        )
    rows = await db.scalars(statement.order_by(Conversation.updated_at.desc()))
    return ConversationListResponse(
        conversations=[
            ConversationSummary(
                id=item.id,
                title=item.title,
                updated_at=item.updated_at.isoformat(),
            )
            for item in rows
        ]
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreateRequest,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> ConversationSummary:
    conversation = Conversation(user_id=current.user.id, title=payload.title.strip())
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        updated_at=conversation.updated_at.isoformat(),
    )


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> ConversationMessagesResponse:
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current.user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = await db.scalars(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation.id)
        .order_by(ConversationMessage.created_at)
    )
    return ConversationMessagesResponse(
        messages=[
            ConversationMessageResponse(
                id=item.id,
                role=item.role,
                content=item.content,
                audit=item.audit,
                created_at=item.created_at.isoformat(),
            )
            for item in messages
        ]
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> None:
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current.user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await db.delete(conversation)
    await db.commit()
