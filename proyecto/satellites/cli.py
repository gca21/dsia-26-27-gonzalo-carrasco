from __future__ import annotations

import argparse
import logging
from pathlib import Path

from satellites.ai.client import AIClientError, get_client
from satellites.config import Settings
from satellites.features import DataTransformationError
from satellites.loader import DataLoadError
from satellites.logging_setup import setup_logging
from satellites.pipeline import GateError, PipelineConfig, SatellitesPipeline
from satellites.predict import PredictionError
from satellites.train import TrainingError
from satellites.validator import DataValidationError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_GATE = 1  # Bad data or a model that is not good enough
EXIT_CONFIG = 2  # Broken configuration or input


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="satellites", description="Satellites classification pipeline"
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw SATCAT csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Artifacts folder")
    parser.add_argument("--max-error-rate", type=float, default=0.1,
                        help="Data gate: max share of invalid rows (default 0.1)")
    parser.add_argument("--min-f1", type=float, default=0.5,
                        help="Model gate: min macro F1 on the test set (default 0.5)")
    parser.add_argument("--explain-top", type=int, default=3,
                        help="Predictions explained by the AI, 0 to disable (default 3)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.output_dir)

    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        max_error_rate=args.max_error_rate,
        min_f1=args.min_f1,
        explain_top=args.explain_top,
    )
    try:
        settings = Settings.from_env()
        client = get_client(
            settings.ai_provider,
            settings.hf_model,
            settings.hf_token,
            settings.hf_inference_provider,
        )
        SatellitesPipeline(config, client).run()
    except GateError:
        return EXIT_GATE  # Already logged by the pipeline
    except (
        DataLoadError, DataValidationError, DataTransformationError,
        TrainingError, PredictionError, AIClientError,
    ) as error:
        logger.error("%s: %s", type(error).__name__, error)
        return EXIT_CONFIG
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())