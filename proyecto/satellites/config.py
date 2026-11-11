from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration read from the environment (.env)"""

    ai_provider: str = "mock"
    hf_token: str | None = None
    hf_model: str | None = None
    hf_inference_provider: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            ai_provider=os.getenv("AI_PROVIDER", "mock"),
            hf_token=os.getenv("HF_TOKEN") or None,
            hf_model=os.getenv("HF_MODEL") or None,
            hf_inference_provider=os.getenv("HF_INFERENCE_PROVIDER") or None,
        )