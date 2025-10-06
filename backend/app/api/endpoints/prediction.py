from fastapi import APIRouter, UploadFile, File
from app.api.models.response_models import PredictionResponse
from app.services.ai_model import LungCancerModel
from app.services.image_processing import ImagePreprocessor


router = APIRouter()
model = LungCancerModel()
preprocessor = ImagePreprocessor()


@router.post("/image", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    preprocessed = preprocessor.preprocess_bytes(image_bytes)
    probability, annotations = model.predict(preprocessed)
    return PredictionResponse(
        probability=probability,
        risk_level=("High" if probability >= 0.66 else "Medium" if probability >= 0.33 else "Low"),
        annotations=annotations,
    )


