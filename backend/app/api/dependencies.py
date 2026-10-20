from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request, params, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.rate_limit import consume
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


def _raise_if_limited(retry_after: int | None) -> None:
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas solicitudes. Espera un momento e inténtalo de nuevo.",
            headers={"Retry-After": str(retry_after)},
        )


def rate_limit(
    scope: str, limit: Callable[[], str], by: Literal["user", "ip"] = "user"
) -> params.Depends:
    """Build a dependency that limits requests per user or per client IP."""

    async def limit_by_user(current: CurrentUserDep) -> None:
        if settings.RATE_LIMIT_ENABLED:
            _raise_if_limited(await consume(scope, current.user.id, limit()))

    async def limit_by_ip(request: Request) -> None:
        if settings.RATE_LIMIT_ENABLED:
            client_ip = request.client.host if request.client else "unknown"
            _raise_if_limited(await consume(scope, client_ip, limit()))

    return Depends(limit_by_user if by == "user" else limit_by_ip)
