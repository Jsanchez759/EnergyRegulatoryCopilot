from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Service health check")
async def health_check() -> dict[str, str]:
    """Return a simple status response to confirm API availability."""
    return {"status": "ok"}
