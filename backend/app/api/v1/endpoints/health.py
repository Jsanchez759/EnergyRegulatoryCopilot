from fastapi import APIRouter
from sqlalchemy import text

from app.api.dependencies import DbSessionDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health_check(db: DbSessionDep) -> dict[str, str]:
    """Return a simple status response to confirm API availability."""
    await db.execute(text("SELECT 1"))
    return {"status": "ok"}
