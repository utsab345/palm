from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimensions: int = Field(default=768, ge=1, le=3072)
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    gemini_timeout_seconds: float = Field(default=30.0, gt=0.0)
    gemini_max_retries: int = Field(default=3, ge=0, le=10)
    gemini_retry_initial_wait: float = Field(default=0.5, ge=0.0)
    gemini_retry_max_wait: float = Field(default=8.0, ge=0.0)
    embedding_batch_size: int = Field(default=100, ge=1, le=1000)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    redis_url: str = "redis://localhost:6379/0"
    chat_memory_ttl_seconds: int = Field(default=86400, ge=60)
    chat_memory_max_messages: int = Field(default=40, ge=2, le=500)

    default_chunk_size: int = Field(default=1000, ge=20, le=8000)
    default_chunk_overlap: int = Field(default=200, ge=0, le=4000)
    max_upload_bytes: int = Field(default=20_000_000, ge=1)

    app_name: str = "Interview RAG Assistant"
    log_level: str = "INFO"

    @property
    def gemini_api_key_value(self) -> str | None:
        if self.gemini_api_key is None:
            return None
        value = self.gemini_api_key.get_secret_value().strip()
        return value or None

    @property
    def qdrant_api_key_value(self) -> str | None:
        if self.qdrant_api_key is None:
            return None
        value = self.qdrant_api_key.get_secret_value().strip()
        return value or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
