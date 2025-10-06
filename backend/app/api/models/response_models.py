from pydantic import BaseModel
from typing import List, Optional


class Annotation(BaseModel):
    x: int
    y: int
    width: int
    height: int
    label: str


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    annotations: Optional[List[Annotation]] = None


class RiskScoreResponse(BaseModel):
    score: float
    level: str


class ReportResponse(BaseModel):
    file_path: str


