import json

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from app.schemas.questions import QuestionRequest, QuestionResponse
from app.services.question_service import QuestionService

router = APIRouter()
question_service = QuestionService()


@router.post("/ask", response_model=QuestionResponse, summary="Answer an energy regulation question")
async def ask_question(
    payload: QuestionRequest,
    x_openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-Api-Key"),
) -> QuestionResponse:
    """Generate a temporary response for a regulatory question."""
    return await question_service.answer(
        question=payload.question,
        history=payload.history,
        api_key_override=x_openrouter_api_key,
    )


@router.post("/ask/stream", summary="Stream an answer for an energy regulation question")
async def ask_question_stream(
    payload: QuestionRequest,
    x_openrouter_api_key: str | None = Header(default=None, alias="X-OpenRouter-Api-Key"),
) -> StreamingResponse:
    """Stream the assistant response as server-sent events."""

    async def event_generator():
        try:
            async for chunk in question_service.stream_answer(
                question=payload.question,
                history=payload.history,
                api_key_override=x_openrouter_api_key,
            ):
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
