from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db import get_db_session
from app.services.agent_service import AgentService
from app.services.auth_service import AuthenticatedUser, current_session
from app.services.document_service import DocumentService

DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(request: Request, db: DbSessionDep) -> AuthenticatedUser:
    token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    authenticated = await current_session(db, token) if token else None
    if authenticated is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inicia sesión para continuar.",
        )
    return authenticated


CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
