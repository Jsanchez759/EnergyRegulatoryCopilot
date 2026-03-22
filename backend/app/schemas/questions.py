from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    """Single chat message used as optional conversation history."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class QuestionRequest(BaseModel):
    """Input payload for user questions related to regulation."""

    question: str = Field(..., min_length=3, max_length=2000, description="Question asked by the user")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="Optional chat history")


class QuestionResponse(BaseModel):
    """API response shape for answered questions."""

    answer: str
    source_hint: str
