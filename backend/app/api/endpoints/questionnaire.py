from fastapi import APIRouter
from app.api.models.request_models import QuestionnaireRequest
from app.api.models.response_models import RiskScoreResponse
from app.services.risk_calculator import RiskCalculator


router = APIRouter()
calculator = RiskCalculator()


@router.post("/risk", response_model=RiskScoreResponse)
def calculate_risk(payload: QuestionnaireRequest) -> RiskScoreResponse:
    score = calculator.calculate(payload)
    level = "High" if score >= 0.66 else ("Medium" if score >= 0.33 else "Low")
    return RiskScoreResponse(score=score, level=level)


