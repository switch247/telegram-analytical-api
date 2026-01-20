from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial News Sentiment Analysis"
    
    # Paths
    ROOT_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = ROOT_DIR / "outputs" / "models"
    
    # Data
    RAW_DATA_FILE: str = "raw_analyst_ratings.csv"
    
    class Config:
        env_file = ".env"

settings = Settings()
