from functools import lru_cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.dependencies import DbSessionDep

router = APIRouter(prefix="/health", tags=["health"])
_BACKEND_ROOT = Path(__file__).resolve().parents[4]


@lru_cache
def expected_schema_revision() -> str | None:
    config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


@router.get("", summary="Liveness probe")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe: database reachable and schema migrated")
async def readiness_check(db: DbSessionDep) -> JSONResponse:
    try:
        current = await db.scalar(text("SELECT version_num FROM alembic_version"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable", "database": "down"})
    expected = expected_schema_revision()
    if current != expected:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "schema": current, "expected_schema": expected},
        )
    return JSONResponse(content={"status": "ok", "schema": current})
