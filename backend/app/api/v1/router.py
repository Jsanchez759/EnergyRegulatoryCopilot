from fastapi import APIRouter

from app.api.v1.endpoints import (
    admin,
    agent,
    auth,
    conversations,
    corpus,
    health,
    market,
    public,
    sources,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(conversations.router)
api_router.include_router(agent.router)
api_router.include_router(sources.router)
api_router.include_router(corpus.router)
api_router.include_router(market.router)
api_router.include_router(public.router)
api_router.include_router(admin.router)
