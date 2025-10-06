from __future__ import annotations

from datetime import datetime
from pathlib import Path
from app.api.models.request_models import ReportRequest


class ReportService:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or "generated_reports")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf(self, payload: ReportRequest) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = self.base_dir / f"report_{timestamp}.txt"
        lines = [
            "Veilo Lung Cancer Detector Report",
            f"Generated at: {datetime.utcnow().isoformat()}Z",
            "",
            f"Probability: {payload.probability:.2f}",
            f"Risk Level: {payload.risk_level}",
            "",
            "Questionnaire:",
            f"  Age: {payload.questionnaire.age}",
            f"  Gender: {payload.questionnaire.gender}",
            f"  Smoking years: {payload.questionnaire.smoking_years}",
            f"  Packs per day: {payload.questionnaire.packs_per_day}",
            f"  Family history: {payload.questionnaire.family_history}",
            f"  Symptoms: {', '.join(payload.questionnaire.symptoms)}",
        ]
        if payload.annotations:
            lines.append("")
            lines.append("Annotations:")
            for a in payload.annotations:
                lines.append(
                    f"  - {a.label} at (x={a.x}, y={a.y}, w={a.width}, h={a.height})"
                )
        file_path.write_text("\n".join(lines), encoding="utf-8")
        return str(file_path)


