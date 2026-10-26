import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    ai_provider: str = os.getenv("AI_PROVIDER", "mock")
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    hf_model: str = os.getenv("HF_MODEL", "owner/name")

    def validate(self) -> None:
        if self.ai_provider == "hf" and not self.hf_token:
            raise ValueError("AI_PROVIDER=hf requiere HF_TOKEN")

settings = Settings()