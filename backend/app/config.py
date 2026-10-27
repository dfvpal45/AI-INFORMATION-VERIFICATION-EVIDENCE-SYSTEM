from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///./aivs.db"
    redis_url: str = "redis://localhost:6379/0"
    google_factcheck_api_key: str = ""
    newsapi_key: str = ""
    anthropic_api_key: str = ""
    enable_transformers: bool = False
    enable_sentence_transformers: bool = False
    tesseract_cmd: str = "tesseract"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
