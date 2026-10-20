from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

Environment = Literal["development", "test", "staging", "production"]


class Settings(BaseSettings):
    """Runtime configuration. Secrets are loaded by Pydantic from backend/.env."""

    ENVIRONMENT: Environment = "development"
    PROJECT_NAME: str = "EnerCol | Regulación energética"
    VERSION: str = "0.3.0"
    API_V1_PREFIX: str = "/api/v1"

    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ALLOWED_ORIGIN_REGEX: str | None = None

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = Field(default="", repr=False)
    OPENROUTER_CHAT_MODEL: str = "anthropic/claude-haiku-5.5"
    OPENROUTER_EMBEDDING_MODEL: str = "perplexity/pplx-embed-v1-0.6b"
    TAVILY_API_KEY: str = Field(default="", repr=False)

    XM_API_URL: str = "https://servapibi.xm.com.co"
    XM_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)
    # Each request covers at most a month, so this bounds the requests behind one tool call.
    MARKET_MAX_RANGE_DAYS: int = Field(default=366, ge=31, le=3660)

    AGENT_MAX_ITERATIONS: int = Field(default=6, ge=1, le=10)
    AGENT_MAX_OUTPUT_TOKENS: int = Field(default=4096, ge=256, le=8192)
    AGENT_CONTEXT_WINDOW: int = Field(default=32768, ge=1024)
    AGENT_TIMEOUT_SECONDS: float = Field(default=90.0, gt=0, le=300)

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://enercol:enercol@localhost:5432/enercol", repr=False
    )
    DATABASE_POOL_SIZE: int = Field(default=5, ge=1, le=50)
    DATABASE_MAX_OVERFLOW: int = Field(default=10, ge=0, le=100)

    SESSION_COOKIE_NAME: str = "enercol_session"
    SESSION_TTL_DAYS: int = Field(default=30, ge=1, le=90)
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"

    # Must match the vector column width created by the migrations.
    EMBEDDING_DIMENSIONS: int = 1024
    RAG_CHUNK_SIZE: int = Field(default=900, ge=200, le=4000)
    RAG_CHUNK_OVERLAP: int = Field(default=160, ge=0, le=1000)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0, le=120)
    EMBEDDING_CONCURRENCY: int = Field(default=4, ge=1, le=16)
    EMBEDDING_MAX_RETRIES: int = Field(default=4, ge=0, le=10)
    MAX_UPLOAD_BYTES: int = Field(default=25 * 1024 * 1024, ge=1024)

    CORPUS_STORAGE_DIR: Path = _BACKEND_ROOT / "storage" / "corpus"
    CORPUS_START_YEAR: int = Field(default=2015, ge=1994)
    CRAWLER_USER_AGENT: str = (
        "EnerColBot/0.1 (+https://github.com/Jsanchez759/EnergyRegulatoryCopilot)"
    )
    CRAWLER_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, le=300)
    CRAWLER_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    CORPUS_CHUNK_CHARS: int = Field(default=2400, ge=500, le=8000)
    OCR_LANGUAGE: str = "spa"
    OCR_MIN_CHARS_PER_PAGE: int = Field(default=200, ge=0)
    # Scanned annexes can run to hundreds of pages; the act itself comes first.
    OCR_MAX_PAGES: int = Field(default=40, ge=1)

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_STORAGE_URI: str = Field(default="memory://", repr=False)
    RATE_LIMIT_AUTH: str = "10/minute;50/hour"
    RATE_LIMIT_AGENT: str = "15/minute;300/day"
    RATE_LIMIT_UPLOAD: str = "20/hour"
    # Each series request may fan out to several XM calls (cached), so this is per user.
    RATE_LIMIT_MARKET: str = "60/minute;2000/day"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] | None = None

    SENTRY_DSN: str = Field(default="", repr=False)
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.0, ge=0.0, le=1.0)

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

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> Self:
        local = self.ENVIRONMENT in {"development", "test"}
        if self.ALLOWED_ORIGIN_REGEX is None and local:
            self.ALLOWED_ORIGIN_REGEX = _LOCALHOST_ORIGIN_REGEX
        if self.LOG_FORMAT is None:
            self.LOG_FORMAT = "console" if local else "json"
        if self.SESSION_COOKIE_SAMESITE == "none" and not self.SESSION_COOKIE_SECURE:
            raise ValueError("SESSION_COOKIE_SAMESITE=none requires SESSION_COOKIE_SECURE=true")
        if self.ENVIRONMENT == "production":
            problems = []
            if not self.SESSION_COOKIE_SECURE:
                problems.append("SESSION_COOKIE_SECURE must be true")
            if not self.OPENROUTER_API_KEY:
                problems.append("OPENROUTER_API_KEY is required")
            if any(
                "localhost" in origin or "127.0.0.1" in origin for origin in self.ALLOWED_ORIGINS
            ):
                problems.append("ALLOWED_ORIGINS must not include localhost")
            if problems:
                raise ValueError("Invalid production settings: " + "; ".join(problems))
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


settings = Settings()
