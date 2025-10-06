from __future__ import annotations

from app.api.models.request_models import QuestionnaireRequest


class RiskCalculator:
    def calculate(self, q: QuestionnaireRequest) -> float:
        score = 0.0
        score += min(q.age / 100.0, 0.3)
        score += min((q.smoking_years * q.packs_per_day) / 100.0, 0.4)
        score += 0.1 if q.family_history else 0.0
        symptom_bonus = min(len(q.symptoms) * 0.05, 0.2)
        score += symptom_bonus
        return max(0.0, min(score, 1.0))


