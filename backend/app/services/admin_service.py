from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    AdminAuditEntry,
    Conversation,
    Document,
    UsageEvent,
    User,
    UserSession,
)
from app.schemas.admin import (
    AdminOverview,
    AdminUser,
    AdminUserDetail,
    AuditEntry,
    BreakdownRow,
    TopUser,
    UsageDay,
    UsageTotals,
)

TIMEZONE = "America/Bogota"
COLOMBIA = ZoneInfo(TIMEZONE)
STATUSES = ("waitlist", "active", "suspended")


def _since(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def _local_day():
    return func.date(func.timezone(TIMEZONE, UsageEvent.created_at))


def _totals_query() -> Select:
    return select(
        func.count(UsageEvent.id),
        func.count(UsageEvent.id).filter(UsageEvent.failed.is_(True)),
        func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
        func.coalesce(func.sum(UsageEvent.prompt_tokens), 0),
        func.coalesce(func.sum(UsageEvent.completion_tokens), 0),
    )


async def _totals(db: AsyncSession, *conditions) -> UsageTotals:
    runs, failed, cost, prompt, completion = (
        await db.execute(_totals_query().where(*conditions))
    ).one()
    return UsageTotals(
        runs=runs,
        failed=failed,
        cost_usd=round(float(cost), 4),
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
    )


async def _daily(db: AsyncSession, days: int, *conditions) -> list[UsageDay]:
    day = _local_day()
    rows = await db.execute(
        select(
            day,
            func.count(UsageEvent.id),
            func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
            func.count(func.distinct(UsageEvent.user_id)),
        )
        .where(UsageEvent.created_at >= _since(days), *conditions)
        .group_by(day)
    )
    found = {row[0]: row for row in rows}
    today = datetime.now(COLOMBIA).date()
    result = []
    for offset in range(days - 1, -1, -1):
        current = today - timedelta(days=offset)
        row = found.get(current)
        result.append(
            UsageDay(
                day=current,
                runs=row[1] if row else 0,
                cost_usd=round(float(row[2]), 4) if row else 0.0,
                users=row[3] if row else 0,
            )
        )
    return result


def _usage_30d_subquery():
    return (
        select(
            UsageEvent.user_id.label("user_id"),
            func.count(UsageEvent.id).label("runs"),
            func.coalesce(func.sum(UsageEvent.cost_usd), 0.0).label("cost"),
        )
        .where(UsageEvent.created_at >= _since(30))
        .group_by(UsageEvent.user_id)
        .subquery()
    )


def _admin_user(user: User, runs: int | None, cost: float | None) -> AdminUser:
    return AdminUser(
        id=user.id,
        display_name=user.display_name,
        email=user.email,
        role=user.role,
        status=user.status,
        company=user.company,
        profile=user.profile,
        created_at=user.created_at,
        approved_at=user.approved_at,
        last_login_at=user.last_login_at,
        runs_30d=runs or 0,
        cost_30d=round(float(cost or 0.0), 4),
    )


def _audit_entry(entry: AdminAuditEntry) -> AuditEntry:
    return AuditEntry(
        id=entry.id,
        actor_email=entry.actor_email,
        action=entry.action,
        target_user_id=entry.target_user_id,
        target_email=entry.target_email,
        detail=entry.detail or {},
        created_at=entry.created_at,
    )


async def list_users(
    db: AsyncSession,
    query: str = "",
    status: str | None = None,
    role: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, dict[str, int], list[AdminUser]]:
    filters = []
    if query.strip():
        pattern = f"%{query.strip()}%"
        filters.append(
            or_(
                User.email.ilike(pattern),
                User.display_name.ilike(pattern),
                User.company.ilike(pattern),
            )
        )
    counts_rows = await db.execute(
        select(User.status, func.count(User.id)).where(*filters).group_by(User.status)
    )
    counts = dict.fromkeys(STATUSES, 0) | {row[0]: row[1] for row in counts_rows}
    counts["admin"] = (
        await db.scalar(select(func.count(User.id)).where(User.role == "admin", *filters)) or 0
    )
    if status:
        filters.append(User.status == status)
    if role:
        filters.append(User.role == role)
    total = await db.scalar(select(func.count(User.id)).where(*filters)) or 0
    usage = _usage_30d_subquery()
    rows = await db.execute(
        select(User, usage.c.runs, usage.c.cost)
        .outerjoin(usage, usage.c.user_id == User.id)
        .where(*filters)
        # Waiting accounts are the ones that need a decision, so they come first.
        .order_by(case((User.status == "waitlist", 0), else_=1), User.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return total, counts, [_admin_user(user, runs, cost) for user, runs, cost in rows]


async def user_detail(db: AsyncSession, user: User) -> AdminUserDetail:
    usage_30d = await _totals(
        db, UsageEvent.user_id == user.id, UsageEvent.created_at >= _since(30)
    )
    conversations = await db.scalar(
        select(func.count(Conversation.id)).where(Conversation.user_id == user.id)
    )
    documents = await db.scalar(select(func.count(Document.id)).where(Document.user_id == user.id))
    sessions = await db.scalar(
        select(func.count(UserSession.id)).where(
            UserSession.user_id == user.id, UserSession.expires_at > datetime.now(UTC)
        )
    )
    audit = await db.scalars(
        select(AdminAuditEntry)
        .where(AdminAuditEntry.target_user_id == user.id)
        .order_by(AdminAuditEntry.created_at.desc())
        .limit(20)
    )
    base = _admin_user(user, usage_30d.runs, usage_30d.cost_usd)
    return AdminUserDetail(
        **base.model_dump(),
        use_case=user.use_case,
        conversations=conversations or 0,
        documents=documents or 0,
        active_sessions=sessions or 0,
        usage_30d=usage_30d,
        usage_all_time=await _totals(db, UsageEvent.user_id == user.id),
        daily=await _daily(db, 30, UsageEvent.user_id == user.id),
        audit=[_audit_entry(entry) for entry in audit],
    )


async def overview(db: AsyncSession) -> AdminOverview:
    status_rows = await db.execute(select(User.status, func.count(User.id)).group_by(User.status))
    users = dict.fromkeys(STATUSES, 0) | {row[0]: row[1] for row in status_rows}
    users["total"] = sum(users[status] for status in STATUSES)
    users["admin"] = await db.scalar(select(func.count(User.id)).where(User.role == "admin")) or 0
    users["new_7d"] = (
        await db.scalar(select(func.count(User.id)).where(User.created_at >= _since(7))) or 0
    )
    users["using_7d"] = (
        await db.scalar(
            select(func.count(func.distinct(UsageEvent.user_id))).where(
                UsageEvent.created_at >= _since(7)
            )
        )
        or 0
    )

    async def breakdown(column) -> list[BreakdownRow]:
        rows = await db.execute(
            select(column, func.count(UsageEvent.id), func.sum(UsageEvent.cost_usd))
            .where(UsageEvent.created_at >= _since(30))
            .group_by(column)
            .order_by(func.sum(UsageEvent.cost_usd).desc())
        )
        return [
            BreakdownRow(name=name, runs=runs, cost_usd=round(float(cost or 0), 4))
            for name, runs, cost in rows
        ]

    cost = func.sum(UsageEvent.cost_usd)
    top_rows = await db.execute(
        select(UsageEvent.user_id, User.display_name, User.email, func.count(UsageEvent.id), cost)
        .outerjoin(User, User.id == UsageEvent.user_id)
        .where(UsageEvent.created_at >= _since(30))
        .group_by(UsageEvent.user_id, User.display_name, User.email)
        .order_by(cost.desc())
        .limit(8)
    )
    return AdminOverview(
        users=users,
        usage_7d=await _totals(db, UsageEvent.created_at >= _since(7)),
        usage_30d=await _totals(db, UsageEvent.created_at >= _since(30)),
        daily=await _daily(db, 30),
        by_feature=await breakdown(UsageEvent.feature),
        by_model=await breakdown(UsageEvent.model),
        top_users=[
            TopUser(
                id=user_id,
                display_name=name or "Cuenta eliminada",
                email=email or "",
                runs=runs,
                cost_usd=round(float(total or 0), 4),
            )
            for user_id, name, email, runs, total in top_rows
        ],
    )


async def audit_log(db: AsyncSession, limit: int = 100) -> list[AuditEntry]:
    entries = await db.scalars(
        select(AdminAuditEntry).order_by(AdminAuditEntry.created_at.desc()).limit(limit)
    )
    return [_audit_entry(entry) for entry in entries]


def record_action(
    db: AsyncSession, actor: User, action: str, target: User, **detail: object
) -> None:
    db.add(
        AdminAuditEntry(
            actor_id=actor.id,
            actor_email=actor.email,
            action=action,
            target_user_id=target.id,
            target_email=target.email,
            detail=detail,
        )
    )


def status_action(previous: str, new: str) -> str:
    if new == "active":
        return "approve" if previous == "waitlist" else "reactivate"
    if new == "suspended":
        return "suspend"
    return "move_to_waitlist"
