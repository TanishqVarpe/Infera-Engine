from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class EntityCard(BaseModel):
    name: str
    entity_type: Literal["person", "organization", "policy", "event", "product", "other"]
    jurisdiction: str
    role_or_context: str
    disambiguation_note: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    resolution_stage: Literal["provisional", "reconciled"] = "provisional"


class EvidenceFactor(BaseModel):
    factor: Annotated[str, Field(max_length=120)]
    supports_forecast: bool  # True if makes THIS scenario's claim more likely
    strength: Literal["strong", "moderate", "weak"]
    reason: Annotated[str, Field(max_length=180)]


class ScoreComponents(BaseModel):
    factors: list[EvidenceFactor] = Field(min_length=3, max_length=4)


class Scenario(BaseModel):
    scenario_name: Annotated[str, Field(max_length=80)]
    time_horizon: Literal["3-6mo", "6-12mo", "1-3yr"]
    forecast_statement: Annotated[str, Field(max_length=500)]
    inferred_from: list[Annotated[str, Field(max_length=120)]] = Field(max_length=3)
    evidence_for: list[Annotated[str, Field(max_length=200)]] = Field(max_length=3)
    evidence_against: list[Annotated[str, Field(max_length=200)]] = Field(max_length=3)
    score_components: ScoreComponents
    score: float = 0.0
    probability_label: Literal["Low", "Medium", "High"] = "Low"
    probability_reason: str = ""


class ScenarioSet(BaseModel):
    scenarios: list[Scenario] = Field(min_length=3, max_length=3)


class BaseRateEstimate(BaseModel):
    reference_class: str
    estimated_base_rate_pct: int = Field(ge=0, le=100)
    base_rate_reasoning: str
    comparable_cases: list[str] = []


class InferaState(TypedDict):
    topic: str
    entity_card: dict | None
    base_rate: dict | None
    analysis: str
    research: str
    uncertainties: str
    scenarios_text: str
    structured_scenarios: dict | None
    current_step: str
