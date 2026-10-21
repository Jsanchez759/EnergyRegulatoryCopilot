from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.api.dependencies import AdminDep, DbSessionDep, get_current_admin
from app.models import User, utc_now
from app.schemas.admin import (
    AccountStatus,
    AdminOverview,
    AdminUser,
    AdminUserDetail,
    AdminUserList,
    AdminUserUpdate,
    AuditLog,
    Role,
)
from app.services import admin_service
from app.services.auth_service import revoke_sessions

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_admin)])


async def target_user(db: DbSessionDep, user_id: str) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    return user


def _forbid_self(admin: AdminDep, target: User, what: str) -> None:
    # Keeps an admin from locking themselves out; another admin has to do it.
    if admin.user.id == target.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"No puedes {what} tu propia cuenta."
        )


@router.get("/overview")
async def get_overview(db: DbSessionDep) -> AdminOverview:
    return await admin_service.overview(db)


@router.get("/users")
async def list_users(
    db: DbSessionDep,
    q: Annotated[str, Query(max_length=120)] = "",
    status_filter: Annotated[AccountStatus | None, Query(alias="status")] = None,
    role: Role | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminUserList:
    total, counts, users = await admin_service.list_users(db, q, status_filter, role, limit, offset)
    return AdminUserList(total=total, counts=counts, users=users)


@router.get("/users/{user_id}")
async def get_user(user_id: str, db: DbSessionDep) -> AdminUserDetail:
    target = await target_user(db, user_id)
    return await admin_service.user_detail(db, target)


@router.patch("/users/{user_id}")
async def update_user(
    payload: AdminUserUpdate, user_id: str, admin: AdminDep, db: DbSessionDep
) -> AdminUser:
    target = await target_user(db, user_id)
    _forbid_self(admin, target, "cambiar el estado o el rol de")
    if payload.status is not None and payload.status != target.status:
        admin_service.record_action(
            db,
            admin.user,
            admin_service.status_action(target.status, payload.status),
            target,
            previous=target.status,
        )
        if payload.status == "active" and target.approved_at is None:
            target.approved_at = utc_now()
        if payload.status != "active":
            await revoke_sessions(db, target.id)
        target.status = payload.status
    if payload.role is not None and payload.role != target.role:
        admin_service.record_action(
            db, admin.user, "promote" if payload.role == "admin" else "demote", target
        )
        target.role = payload.role
    await db.commit()
    detail = await admin_service.user_detail(db, target)
    return AdminUser(**detail.model_dump(include=set(AdminUser.model_fields)))


@router.post("/users/{user_id}/revoke-sessions", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_user_sessions(user_id: str, admin: AdminDep, db: DbSessionDep) -> Response:
    target = await target_user(db, user_id)
    await revoke_sessions(db, target.id)
    admin_service.record_action(db, admin.user, "revoke_sessions", target)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, admin: AdminDep, db: DbSessionDep) -> Response:
    target = await target_user(db, user_id)
    _forbid_self(admin, target, "eliminar")
    admin_service.record_action(
        db, admin.user, "delete", target, previous=target.status, name=target.display_name
    )
    await db.delete(target)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/audit")
async def get_audit_log(
    db: DbSessionDep, limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> AuditLog:
    return AuditLog(entries=await admin_service.audit_log(db, limit))
