from typing import TypedDict, List, Optional, Literal
from pydantic import AliasChoices, BaseModel, Field, model_validator


class EntityCard(BaseModel):
    name: str
    entity_type: Literal["person", "organization", "policy", "event", "product", "other"] = Field(
        validation_alias=AliasChoices("entity_type", "type")
    )
    jurisdiction: str
    role_or_context: str = Field(
        validation_alias=AliasChoices("role_or_context", "role")
    )
    disambiguation_note: str = Field(
        default="",
        validation_alias=AliasChoices("disambiguation_note", "note"),
    )
    resolution_stage: Literal["provisional", "reconciled"] = "provisional"

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, value):
        if not isinstance(value, dict):
            return value

        aliases = {
            "name": "name",
            "entity_type": "entity_type",
            "type": "entity_type",
            "jurisdiction": "jurisdiction",
            "role_or_context": "role_or_context",
            "role": "role_or_context",
            "disambiguation_note": "disambiguation_note",
            "note": "disambiguation_note",
            "resolution_stage": "resolution_stage",
        }
        normalized = {}
        for key, field_value in value.items():
            canonical_key = aliases.get(str(key).strip().lower())
            if canonical_key:
                normalized[canonical_key] = field_value
        return normalized


class EvidenceFactor(BaseModel):
    factor: str = Field(validation_alias=AliasChoices("factor", "description"))
    supports_forecast: bool  # True if makes THIS scenario's claim more likely
    strength: Literal["strong", "moderate", "weak"]
    reason: str = ""


class ScoreComponents(BaseModel):
    factors: List[EvidenceFactor] = Field(min_length=3, max_length=6)


class Scenario(BaseModel):
    scenario_name: str
    time_horizon: Literal["3-6mo", "6-12mo", "1-3yr"]
    forecast_statement: str
    inferred_from: List[str]
    evidence_for: List[str]
    evidence_against: List[str]
    score_components: ScoreComponents


class ScenarioSet(BaseModel):
    scenarios: List[Scenario] = Field(
        min_length=3,
        max_length=3,
        validation_alias=AliasChoices("scenarios", "scenario_set")
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_scenario_set(cls, value):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and (
                    "scenarios" in item or "scenario_set" in item
                ):
                    return item
        return value


class BaseRateEstimate(BaseModel):
    reference_class: str
    estimated_base_rate_pct: int = Field(ge=0, le=100)
    base_rate_reasoning: str = "Not provided by the model."
    comparable_cases: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_reference_class(cls, value):
        if isinstance(value, dict):
            value = value.copy()
            if not value.get("reference_class"):
                direction = value.get("reference_class_direction")
                scale = value.get("scale")
                if direction or scale:
                    value["reference_class"] = " ".join(
                        part for part in (direction, scale) if part
                    )
        return value


class InferaState(TypedDict):
    topic: str
    entity_card: Optional[dict]
    base_rate: Optional[dict]
    analysis: str
    research: str
    uncertainties: str
    scenarios_text: str
    structured_scenarios: Optional[dict]
    current_step: str