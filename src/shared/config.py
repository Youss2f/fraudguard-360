"""
FraudGuard-360 Configuration Management
========================================

Centralized configuration using pydantic-settings.
All secrets loaded strictly from environment variables.

Author: FraudGuard-360 Platform Team
License: MIT
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")
    
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="fraudguard")
    user: str = Field(default="fraudguard")
    password: SecretStr = Field(default=SecretStr("changeme"))
    pool_size: int = Field(default=10)
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.name}"
    
    @property
    def safe_url(self) -> str:
        return f"postgresql://{self.user}:****@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")
    
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[SecretStr] = Field(default=None)
    socket_timeout: int = Field(default=5)
    
    @property
    def url(self) -> str:
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class KafkaSettings(BaseSettings):
    """Kafka configuration."""
    
    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")
    
    bootstrap_servers: str = Field(default="localhost:9092")
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: Optional[str] = Field(default=None)
    sasl_username: Optional[str] = Field(default=None)
    sasl_password: Optional[SecretStr] = Field(default=None)
    
    raw_transactions_topic: str = Field(default="raw-transactions")
    scored_transactions_topic: str = Field(default="scored-transactions")
    consumer_group: str = Field(default="fraudguard-consumer-group")
    
    auto_offset_reset: str = Field(default="earliest")
    max_poll_records: int = Field(default=100)
    session_timeout_ms: int = Field(default=30000)
    acks: str = Field(default="all")
    retries: int = Field(default=3)
    
    @property
    def servers_list(self) -> List[str]:
        return [s.strip() for s in self.bootstrap_servers.split(",")]


class JWTSettings(BaseSettings):
    """JWT configuration."""
    
    model_config = SettingsConfigDict(env_prefix="JWT_", extra="ignore")
    
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)


class ServiceSettings(BaseSettings):
    """Service configuration."""
    
    model_config = SettingsConfigDict(env_prefix="SERVICE_", extra="ignore")
    
    name: str = Field(default="fraudguard-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    request_timeout: int = Field(default=30)


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    
    ml_service_url: str = Field(default="http://ml-service:8000")
    risk_scoring_url: str = Field(default="http://risk-scoring-service:8001")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
