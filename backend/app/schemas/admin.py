from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, model_validator

AccountStatus = Literal["waitlist", "active", "suspended"]
Role = Literal["user", "admin"]


class UsageTotals(BaseModel):
    runs: int
    failed: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class UsageDay(BaseModel):
    day: date
    runs: int
    cost_usd: float
    users: int


class AdminUser(BaseModel):
    id: str
    display_name: str
    email: str
    role: Role
    status: AccountStatus
    company: str | None
    profile: str | None
    created_at: datetime
    approved_at: datetime | None
    last_login_at: datetime | None
    runs_30d: int
    cost_30d: float


class AdminUserList(BaseModel):
    total: int
    counts: dict[str, int]
    users: list[AdminUser]


class AuditEntry(BaseModel):
    id: str
    actor_email: str
    action: str
    target_user_id: str | None
    target_email: str | None
    detail: dict
    created_at: datetime


class AdminUserDetail(AdminUser):
    use_case: str | None
    conversations: int
    documents: int
    active_sessions: int
    usage_30d: UsageTotals
    usage_all_time: UsageTotals
    daily: list[UsageDay]
    audit: list[AuditEntry]


class AdminUserUpdate(BaseModel):
    status: AccountStatus | None = None
    role: Role | None = None

    @model_validator(mode="after")
    def require_change(self) -> "AdminUserUpdate":
        if self.status is None and self.role is None:
            raise ValueError("Indica el estado o el rol que quieres cambiar.")
        return self


class BreakdownRow(BaseModel):
    name: str
    runs: int
    cost_usd: float


class TopUser(BaseModel):
    id: str | None
    display_name: str
    email: str
    runs: int
    cost_usd: float


class AdminOverview(BaseModel):
    users: dict[str, int]
    usage_7d: UsageTotals
    usage_30d: UsageTotals
    daily: list[UsageDay]
    by_feature: list[BreakdownRow]
    by_model: list[BreakdownRow]
    top_users: list[TopUser]


class AuditLog(BaseModel):
    entries: list[AuditEntry]
