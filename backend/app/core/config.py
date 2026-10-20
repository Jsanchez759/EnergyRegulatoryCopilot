from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Secrets are loaded by Pydantic from backend/.env."""

    PROJECT_NAME: str = "EnerCol | Regulación energética"
    VERSION: str = "0.2.0"
    API_V1_PREFIX: str = "/api/v1"

    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ALLOWED_ORIGIN_REGEX: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = Field(default="", repr=False)
    OPENROUTER_CHAT_MODEL: str = "openrouter/free"
    OPENROUTER_EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    TAVILY_API_KEY: str = Field(default="", repr=False)

    AGENT_MAX_ITERATIONS: int = Field(default=4, ge=1, le=8)
    AGENT_MAX_OUTPUT_TOKENS: int = Field(default=4096, ge=256, le=8192)
    AGENT_CONTEXT_WINDOW: int = Field(default=32768, ge=1024)
    AGENT_TIMEOUT_SECONDS: float = Field(default=90.0, gt=0, le=300)

    DATABASE_PATH: Path = _BACKEND_ROOT / "storage" / "enercol.db"
    DATABASE_URL: str | None = None
    SESSION_COOKIE_NAME: str = "enercol_session"
    SESSION_TTL_DAYS: int = Field(default=30, ge=1, le=90)
    SESSION_COOKIE_SECURE: bool = False

    VECTOR_DB_PATH: Path = _BACKEND_ROOT / "storage" / "chroma"
    VECTOR_COLLECTION_NAME: str = "regulatory_documents"
    RAG_CHUNK_SIZE: int = Field(default=900, ge=200, le=4000)
    RAG_CHUNK_OVERLAP: int = Field(default=160, ge=0, le=1000)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=120)
    EMBEDDING_MAX_RETRIES: int = Field(default=4, ge=0, le=10)
    MAX_UPLOAD_BYTES: int = Field(default=25 * 1024 * 1024, ge=1024)
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("RAG_CHUNK_OVERLAP")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("RAG_CHUNK_SIZE", 900)
        if value >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return value


settings = Settings()
