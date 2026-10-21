from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.schemas.market import MarketIndicator


class PublicSource(BaseModel):
    id: str
    name: str
    norms: int


class PublicConsultation(BaseModel):
    label: str
    title: str
    source: str
    comment_end: date
    url: str


class PublicNorm(BaseModel):
    label: str
    title: str
    source: str
    issued_on: date | None
    url: str


class PublicOverview(BaseModel):
    registration_mode: Literal["open", "waitlist"]
    sources: list[PublicSource]
    articles: int
    open_consultations: int
    closing_soon: list[PublicConsultation]
    latest: list[PublicNorm]
    market: list[MarketIndicator]
