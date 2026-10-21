from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import CurrentUserDep, DbSessionDep, rate_limit
from app.core.config import settings
from app.models import User, UserSession
from app.schemas.auth import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    LoginRequest,
    ProfileUpdateRequest,
    RegisterRequest,
    UserResponse,
)
from app.services.auth_service import (
    DuplicateEmailError,
    authenticate,
    digest_token,
    hash_password,
    issue_session,
    register_user,
    revoke_sessions,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["authentication"])
auth_rate_limit = rate_limit("auth", lambda: settings.RATE_LIMIT_AUTH, by="ip")
INACTIVE_MESSAGES = {
    "waitlist": (
        "Tu cuenta está en la lista de espera de la beta. Te avisaremos cuando tengas acceso."
    ),
    "suspended": "Tu cuenta está suspendida. Escríbenos si crees que es un error.",
}


def user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        display_name=user.display_name,
        email=user.email,
        role=user.role,
        status=user.status,
        created_at=user.created_at.isoformat(),
    )


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        settings.SESSION_COOKIE_NAME,
        token,
        max_age=settings.SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        settings.SESSION_COOKIE_NAME,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )


@router.post("/register", status_code=status.HTTP_201_CREATED, dependencies=[auth_rate_limit])
async def register(
    payload: RegisterRequest,
    response: Response,
    db: DbSessionDep,
) -> UserResponse:
    try:
        user = await register_user(
            db,
            payload.display_name,
            str(payload.email),
            payload.password,
            company=payload.company,
            profile=payload.profile,
            use_case=payload.use_case,
        )
    except DuplicateEmailError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta con este correo electrónico.",
        ) from exc
    # Waitlisted accounts get no session; the response tells the page to show the waitlist note.
    if user.status == "active":
        set_session_cookie(response, await issue_session(db, user))
    return user_response(user)


@router.post("/login", dependencies=[auth_rate_limit])
async def login(
    payload: LoginRequest,
    response: Response,
    db: DbSessionDep,
) -> UserResponse:
    user = await authenticate(db, str(payload.email), payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El correo electrónico o la contraseña no son correctos.",
        )
    # Only someone who knows the password learns the account's state.
    if user.status in INACTIVE_MESSAGES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=INACTIVE_MESSAGES[user.status]
        )
    set_session_cookie(response, await issue_session(db, user))
    return user_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, db: DbSessionDep) -> Response:
    token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if token:
        session = await db.scalar(
            select(UserSession).where(UserSession.token_digest == digest_token(token))
        )
        if session is not None:
            await db.delete(session)
            await db.commit()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(response)
    return response


@router.get("/me")
async def get_profile(current: CurrentUserDep) -> UserResponse:
    return user_response(current.user)


@router.patch("/me")
async def update_profile(
    payload: ProfileUpdateRequest,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> UserResponse:
    user = current.user
    if payload.display_name is not None:
        user.display_name = payload.display_name.strip()
    if payload.email is not None:
        email = str(payload.email).strip().casefold()
        if email != user.email:
            if not payload.current_password or not verify_password(
                payload.current_password, user.password_hash
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Escribe tu contraseña actual para cambiar el correo.",
                )
            existing = await db.scalar(select(User.id).where(User.email == email))
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Ya existe una cuenta con este correo electrónico.",
                )
            user.email = email
    await db.commit()
    await db.refresh(user)
    return user_response(user)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT, dependencies=[auth_rate_limit])
async def change_password(
    payload: ChangePasswordRequest,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> Response:
    if not verify_password(payload.current_password, current.user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña actual no es correcta.",
        )
    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Elige una contraseña distinta a la actual.",
        )
    current.user.password_hash = hash_password(payload.new_password)
    await revoke_sessions(db, current.user.id)
    await db.commit()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    set_session_cookie(response, await issue_session(db, current.user))
    return response


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    payload: DeleteAccountRequest,
    db: DbSessionDep,
    current: CurrentUserDep,
) -> Response:
    if not verify_password(payload.password, current.user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña no es correcta.",
        )
    await db.delete(current.user)
    await db.commit()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(response)
    return response
