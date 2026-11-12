from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration read from the environment (.env)"""

    spacetrack_user: str | None = None
    spacetrack_pass: str | None = None
    ai_provider: str = "mock"
    hf_token: str | None = None
    hf_model: str | None = None
    hf_inference_provider: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            spacetrack_user=os.getenv("SPACETRACK_USER") or None,
            spacetrack_pass=os.getenv("SPACETRACK_PASS") or None,
            ai_provider=os.getenv("AI_PROVIDER", "mock"),
            hf_token=os.getenv("HF_TOKEN") or None,
            hf_model=os.getenv("HF_MODEL") or None,
            hf_inference_provider=os.getenv("HF_INFERENCE_PROVIDER") or None,
        )