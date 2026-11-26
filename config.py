import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-2026")
    
    # File Upload Settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))  # 16 MB max limit
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {"pdf"}

    # Database Configuration (SQLite)
    DATABASE_DIR = BASE_DIR / "database"
    DATABASE_PATH = DATABASE_DIR / "study_assistant.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AI Model Cache & Summarization Settings
    MODELS_DIR = BASE_DIR / "models"
    HF_CACHE_DIR = MODELS_DIR / "cache"
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
    DEFAULT_SUMMARY_TYPE = os.getenv("DEFAULT_SUMMARY_TYPE", "standard")

    # Phase 4: RAG Vector Store & Chatbot Settings
    VECTOR_STORE_DIR = BASE_DIR / "database" / "vector_store"
    RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RAG_QA_MODEL = os.getenv("RAG_QA_MODEL", "google/flan-t5-small")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
    RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.28"))
    RAG_CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "120"))
    RAG_CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "30"))

    # Point HuggingFace cache to D: drive (MODELS_DIR/cache) to prevent C: disk space exhaustion
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    # Silence Windows symlink warning for HuggingFace Hub
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Constrain OpenBLAS / OMP thread pools to prevent memory allocation exhaustion on Windows
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    DATABASE_PATH = Config.DATABASE_DIR / "test_study_assistant.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
