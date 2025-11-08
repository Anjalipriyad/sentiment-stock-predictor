"""
Configuration module for Sentiment Stock Predictor backend.
Loads all environment variables and constants using Pydantic BaseSettings.
"""

from pydantic_settings import BaseSettings


from functools import lru_cache
import os


class Settings(BaseSettings):
    # -----------------------------
    # Application Settings
    # -----------------------------
    APP_NAME: str = "Sentiment Stock Predictor"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "info"

    # -----------------------------
    # Database Configuration
    # -----------------------------
    DB_URL: str = "sqlite:///./app.db"  # Default to SQLite if Postgres not configured

    # -----------------------------
    # External API Keys
    # -----------------------------
    NEWS_API_KEY: str | None = None
    TWITTER_BEARER_TOKEN: str | None = None
    ALPHA_VANTAGE_KEY: str | None = None

    # -----------------------------
    # Model Paths
    # -----------------------------
    SENTIMENT_MODEL_PATH: str = "app/ml_models/sentiment/finbert/"
    PRICE_MODEL_PATH: str = "app/ml_models/price_predictor/model.pkl"

    # -----------------------------
    # Frontend / Client Settings
    # -----------------------------
    FRONTEND_URL: str = "http://localhost:3000"

    # -----------------------------
    # Data Directories
    # -----------------------------
    RAW_DATA_DIR: str = "data/raw/"
    PROCESSED_DATA_DIR: str = "data/processed/"
    PREDICTION_DATA_DIR: str = "data/predictions/"

    class Config:
        env_file = ".env"        # Automatically load from .env
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader so that the environment variables are read only once.
    Usage:
        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Initialize settings on import
settings = get_settings()
