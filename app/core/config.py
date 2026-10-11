from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vision API MLOps"
    VERSION: str = "1.0.0"
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    DEBUG_MODE: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
