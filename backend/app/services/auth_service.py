import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pwdlib import PasswordHash
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import User, UserSession

password_hasher = PasswordHash.recommended()
DUMMY_PASSWORD_HASH = password_hasher.hash("invalid-account-password-placeholder")


class DuplicateEmailError(ValueError):
    """Raised when an account already uses the requested email address."""


class InvalidCredentialsError(ValueError):
    """Raised when a password does not match the account."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user: User
    session: UserSession
    token: str


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_hasher.verify(password, password_hash)


def digest_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def issue_session(db: AsyncSession, user: User) -> str:
    token = secrets.token_urlsafe(48)
    user.last_login_at = datetime.now(UTC)
    await db.execute(
        delete(UserSession).where(
            UserSession.user_id == user.id, UserSession.expires_at <= datetime.now(UTC)
        )
    )
    db.add(
        UserSession(
            user_id=user.id,
            token_digest=digest_token(token),
            expires_at=datetime.now(UTC) + timedelta(days=settings.SESSION_TTL_DAYS),
        )
    )
    await db.commit()
    return token


async def register_user(
    db: AsyncSession,
    display_name: str,
    email: str,
    password: str,
    *,
    company: str | None = None,
    profile: str | None = None,
    use_case: str | None = None,
) -> User:
    normalized_email = email.strip().casefold()
    existing = await db.scalar(select(User.id).where(User.email == normalized_email))
    if existing:
        raise DuplicateEmailError("Ya existe una cuenta con este correo electrónico.")
    waitlist = settings.REGISTRATION_MODE == "waitlist"
    user = User(
        display_name=display_name.strip(),
        email=normalized_email,
        password_hash=hash_password(password),
        status="waitlist" if waitlist else "active",
        approved_at=None if waitlist else datetime.now(UTC),
        company=company,
        profile=profile,
        use_case=use_case,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate(db: AsyncSession, email: str, password: str) -> User | None:
    normalized_email = email.strip().casefold()
    user = await db.scalar(select(User).where(User.email == normalized_email))
    password_matches = verify_password(
        password, user.password_hash if user is not None else DUMMY_PASSWORD_HASH
    )
    if user is None or not password_matches:
        return None
    return user


async def current_session(db: AsyncSession, token: str) -> AuthenticatedUser | None:
    session = await db.scalar(
        select(UserSession).where(UserSession.token_digest == digest_token(token))
    )
    if session is None:
        return None
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        await db.delete(session)
        await db.commit()
        return None
    user = await db.get(User, session.user_id)
    if user is None or user.status != "active":
        return None
    return AuthenticatedUser(user=user, session=session, token=token)


async def revoke_sessions(db: AsyncSession, user_id: str) -> None:
    await db.execute(delete(UserSession).where(UserSession.user_id == user_id))
