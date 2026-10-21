import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent

from app.db import SessionFactory
from app.models import UsageEvent

logger = logging.getLogger(__name__)


@dataclass
class UsageMeter:
    feature: str
    model: str
    started: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, usage: object) -> None:
        self.llm_calls += 1
        self.prompt_tokens += int(_field(usage, "prompt_tokens") or 0)
        self.completion_tokens += int(_field(usage, "completion_tokens") or 0)
        # OpenRouter reports the billed cost in USD next to the token counts.
        self.cost_usd += float(_field(usage, "cost") or 0.0)

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)


# Agent steps run in tasks that copy the request's context, so they all see the same meter.
_current_meter: ContextVar[UsageMeter | None] = ContextVar("usage_meter", default=None)


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


class _UsageHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "EnerColUsageHandler"

    def handle(self, event: BaseEvent, **kwargs: object) -> None:
        meter = _current_meter.get()
        if meter is None or not isinstance(event, LLMChatEndEvent) or event.response is None:
            return
        usage = _field(event.response.raw, "usage")
        if usage is not None:
            meter.add(usage)


_handler = _UsageHandler()
_dispatcher = get_dispatcher()
if not any(isinstance(item, _UsageHandler) for item in _dispatcher.event_handlers):
    _dispatcher.add_event_handler(_handler)


def start_metering(feature: str, model: str) -> UsageMeter:
    """Meter every model call made from the current request from here on."""
    meter = UsageMeter(feature=feature, model=model)
    _current_meter.set(meter)
    return meter


async def record_usage(meter: UsageMeter, user_id: str, failed: bool) -> None:
    try:
        async with SessionFactory() as session:
            session.add(
                UsageEvent(
                    user_id=user_id,
                    feature=meter.feature,
                    model=meter.model,
                    llm_calls=meter.llm_calls,
                    prompt_tokens=meter.prompt_tokens,
                    completion_tokens=meter.completion_tokens,
                    cost_usd=round(meter.cost_usd, 6),
                    duration_ms=meter.duration_ms,
                    failed=failed,
                )
            )
            await session.commit()
    except Exception:
        # Losing one usage row must never break the answer the user already received.
        logger.exception("could not record assistant usage")
