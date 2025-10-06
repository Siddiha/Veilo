from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionnaireRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    gender: str
    smoking_years: int = Field(ge=0, le=80)
    packs_per_day: float = Field(ge=0.0, le=5.0)
    symptoms: List[str] = []
    family_history: bool = False


class Annotation(BaseModel):
    x: int
    y: int
    width: int
    height: int
    label: str


class ReportRequest(BaseModel):
    probability: float
    risk_level: str
    questionnaire: QuestionnaireRequest
    annotations: Optional[List[Annotation]] = None


