from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.prediction import router as prediction_router
from app.api.endpoints.questionnaire import router as questionnaire_router
from app.api.endpoints.reports import router as reports_router


def create_app() -> FastAPI:
    app = FastAPI(title="Veilo Lung Cancer Detector API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(prediction_router, prefix="/api/v1/predict", tags=["predict"])
    app.include_router(questionnaire_router, prefix="/api/v1/questionnaire", tags=["questionnaire"])
    app.include_router(reports_router, prefix="/api/v1/reports", tags=["reports"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


