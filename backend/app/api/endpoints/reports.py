from fastapi import APIRouter
from app.api.models.request_models import ReportRequest
from app.api.models.response_models import ReportResponse
from app.services.report_service import ReportService


router = APIRouter()
report_service = ReportService()


@router.post("/generate", response_model=ReportResponse)
def generate_report(payload: ReportRequest) -> ReportResponse:
    file_path = report_service.generate_pdf(payload)
    return ReportResponse(file_path=file_path)


