from fastapi import APIRouter

from app.api.v1.endpoints import health, questions, rag

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(questions.router, prefix="/questions", tags=["questions"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
