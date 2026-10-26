import logging
import sys
from pathlib import Path

def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )