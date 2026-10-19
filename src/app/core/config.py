from enum import Enum
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class BaseAppSettings(BaseSettings):
    """所有配置类的公共基类，统一指定 .env 路径与 UTF-8 编码。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


class AppSettings(BaseAppSettings):
    APP_NAME: str = "FastAPI Template"
    APP_DESCRIPTION: str | None = None
    APP_VERSION: str | None = "0.1.0"
    LICENSE_NAME: str | None = "MIT"
    CONTACT_NAME: str | None = None
    CONTACT_EMAIL: str | None = None


class LoggingSettings(BaseAppSettings):
    LOG_LEVEL: str = "INFO"
    DB_ECHO: bool = False


class CryptSettings(BaseAppSettings):
    SECRET_KEY: str = "change-me-please-generate-a-random-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7


class DatabaseSettings(BaseAppSettings):
    """数据库配置的标记基类，具体实现见各数据库子类。"""


class SQLiteSettings(DatabaseSettings):
    SQLITE_URI: str = "./sql_app.db"
    SQLITE_SYNC_PREFIX: str = "sqlite:///"
    SQLITE_ASYNC_PREFIX: str = "sqlite+aiosqlite:///"


class MySQLSettings(DatabaseSettings):
    MYSQL_USER: str = "username"
    MYSQL_PASSWORD: str = "password"
    MYSQL_SERVER: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "dbname"
    MYSQL_SYNC_PREFIX: str = "mysql://"
    MYSQL_ASYNC_PREFIX: str = "mysql+aiomysql://"
    MYSQL_URL: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def MYSQL_URI(self) -> str:  # type: ignore[prop-decorator]  # noqa: N802
        return f"{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"


class PostgresSettings(DatabaseSettings):
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "postgres"
    POSTGRES_SYNC_PREFIX: str = "postgresql://"
    POSTGRES_ASYNC_PREFIX: str = "postgresql+asyncpg://"
    POSTGRES_URL: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def POSTGRES_URI(self) -> str:  # type: ignore[prop-decorator]  # noqa: N802
        return (
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


class FirstUserSettings(BaseAppSettings):
    ADMIN_NAME: str = "admin"
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-this-password"


class TestSettings(BaseAppSettings):
    """测试环境预留配置。"""


class RedisCacheSettings(BaseAppSettings):
    REDIS_CACHE_HOST: str = "localhost"
    REDIS_CACHE_PORT: int = 6379

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_CACHE_URL(self) -> str:  # type: ignore[prop-decorator]  # noqa: N802
        return f"redis://{self.REDIS_CACHE_HOST}:{self.REDIS_CACHE_PORT}"


class ClientSideCacheSettings(BaseAppSettings):
    CLIENT_CACHE_MAX_AGE: int = 60


class RedisQueueSettings(BaseAppSettings):
    REDIS_QUEUE_HOST: str = "localhost"
    REDIS_QUEUE_PORT: int = 6379


class RedisRateLimiterSettings(BaseAppSettings):
    REDIS_RATE_LIMIT_HOST: str = "localhost"
    REDIS_RATE_LIMIT_PORT: int = 6379

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_RATE_LIMIT_URL(self) -> str:  # type: ignore[prop-decorator]  # noqa: N802
        return f"redis://{self.REDIS_RATE_LIMIT_HOST}:{self.REDIS_RATE_LIMIT_PORT}"


class DefaultRateLimitSettings(BaseAppSettings):
    DEFAULT_RATE_LIMIT_LIMIT: int = 10
    DEFAULT_RATE_LIMIT_PERIOD: int = 3600


class SkipInfraSettings(BaseAppSettings):
    SKIP_INFRA_ON_START: bool = False


class EnvironmentOption(Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(BaseAppSettings):
    ENVIRONMENT: EnvironmentOption = EnvironmentOption.LOCAL


class EmbeddingModelSettings(BaseAppSettings):
    DASHSCOPE_API_KEY: str | None = None
    DASHSCOPE_MODEL: str | None = None
    DASHSCOPE_RERANK_MODEL: str | None = None


class DeepSeekSettings(BaseAppSettings):
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_MODEL: str | None = None
    DEEPSEEK_BASE_URL: str | None = None


class LocaRagSettings(BaseAppSettings):
    DOCUMENT_DIR: Path | None = None


class VisionModelSettings(BaseAppSettings):
    """千问视觉模型配置。"""

    DASHSCOPE_VL_MODEL: str | None = None



class AliyunOSSSettings(BaseAppSettings):
    """阿里云 OSS 对象存储配置。"""

    OSS_ACCESS_KEY_ID: str | None = None
    OSS_ACCESS_KEY_SECRET: str | None = None
    OSS_BUCKET_NAME: str | None = None
    OSS_ENDPOINT: str | None = None
    OSS_REGION: str | None = None


class Settings(
    AppSettings,
    LoggingSettings,
    PostgresSettings,
    CryptSettings,
    FirstUserSettings,
    TestSettings,
    RedisCacheSettings,
    ClientSideCacheSettings,
    RedisQueueSettings,
    RedisRateLimiterSettings,
    DefaultRateLimitSettings,
    SkipInfraSettings,
    EnvironmentSettings,
    EmbeddingModelSettings,
    DeepSeekSettings,
    LocaRagSettings,
    VisionModelSettings,
    AliyunOSSSettings,
):
    """聚合所有配置的顶层配置对象。"""


settings = Settings()
