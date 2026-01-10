from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtensionConfig(BaseSettings):
    """Shared config loader for extensions with safe env var fallbacks."""

    model_config = SettingsConfigDict(
        env_prefix="EXTENSION_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")


def get_extension_config() -> ExtensionConfig:
    """Safely load extension config with fallbacks."""
    return ExtensionConfig()
