from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = Field(default="gpt-4o", description="The LLM model to use")
    max_steps: int = Field(default=10, description="Max steps for agent execution")
    max_tokens: int = Field(default=512, description="Max tokens per LLM call")

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="console", description="Logging format (json or console)")

    # Add these for OpenRouter
    openrouter_api_key: str = Field(default="", description="OpenRouter API key (sk-or-v1-...)")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()