from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Veilo Lung Cancer Detector API"
    environment: str = "dev"
    model_path: str = "models/trained_models/lung_cancer_model.pth"
    preprocess_path: str = "models/trained_models/preprocessing.pkl"

    class Config:
        env_file = ".env"


settings = Settings()


