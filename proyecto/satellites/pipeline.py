from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datetime import datetime, timezone
from satellites.downloader import DataDownloader
from satellites.ai.client import ModelClient
from satellites.ai.explain import explain_prediction
from satellites.loader import CsvSatellitesRepository
from satellites.predict import SatellitesPredictor
from satellites.train import SatellitesTrainer
from satellites.validator import SatellitesValidator

logger = logging.getLogger(__name__)


class GateError(Exception):
    """Bad data or a model that is not good enough."""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    max_error_rate: float = 0.1  # Max rate of invalid rows
    min_f1: float = 0.5  # Min F1 on the test set
    explain_top: int = 3  # Objects explained by the AI (0 disables it)


class SatellitesPipeline:
    """download -> load -> validate -> data gate -> train -> model gate -> predict -> explain"""

    METRICS_FILE = "metrics.json"
    CLEAN_FILE = "clean_satellites.csv"
    INVALID_FILE = "invalid_satellites.csv"
    PREDICTIONS_FILE = "predictions.csv"

    def __init__(
        self,
        config: PipelineConfig,
        client: ModelClient,
        trainer: SatellitesTrainer | None = None,
        downloader: DataDownloader | None = None,
    ) -> None:
            self.config = config
            self.client = client
            self.trainer = trainer or SatellitesTrainer()
            self.downloader = downloader  # None = use the existing input file

    def run(self) -> dict:
        """Runs every stage. Writes the artifacts in output_dir (overwriting them)
        and returns the metrics. Raises GateError if a gate fails"""

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Pipeline started: %s -> %s", self.config.input_path, output_dir)

        # Download
        if self.downloader is not None:
            rows = self.downloader.download(self.config.input_path)
            download = {
                "downloaded": True,
                "rows": rows,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        else:
            logger.info("Download skipped, using the existing file")
            download = {"downloaded": False}

        # Repository and metrics
        raw = CsvSatellitesRepository(self.config.input_path).load()
        valid, invalid = SatellitesValidator(raw).validate()
        error_rate = len(invalid) / len(raw) if len(raw) else 1.0
        metrics: dict = {
            "download": download,
            "rows_in": len(raw),
            "rows_valid": len(valid),
            "rows_invalid": len(invalid),
            "error_rate": round(error_rate, 4),
        }
        valid.to_csv(output_dir / self.CLEAN_FILE, index=False)
        invalid.to_csv(output_dir / self.INVALID_FILE, index=False)
        logger.info("Validation: %d valid, %d invalid rows", len(valid), len(invalid))

        if error_rate > self.config.max_error_rate:
            self._fail(
                metrics, "data_gate_failed",
                f"Data gate failed: error rate {error_rate:.3f} "
                f"> {self.config.max_error_rate:.3f}",
            )

        # Training
        model_metrics = self.trainer.train(valid, output_dir)
        metrics["model"] = {
            key: model_metrics[key] for key in ("accuracy", "f1_macro", "baseline_f1_macro")
        }
        if model_metrics["f1_macro"] < self.config.min_f1:
            self._fail(
                metrics, "model_gate_failed",
                f"Model gate failed: macro F1 {model_metrics['f1_macro']:.3f} "
                f"< {self.config.min_f1:.3f}",
            )

        metrics["predictions"] = self._predict_unclassified(valid)
        metrics["status"] = "ok"
        self._write_metrics(metrics)
        logger.info("Pipeline finished correctly")
        return metrics

    def _predict_unclassified(self, valid: pd.DataFrame) -> dict:
        """Classifies the objects the catalog does not know,
        which are not used for training, and explains the most confident ones"""

        unclassified = valid[~valid["OBJECT_TYPE"].isin(self.trainer.TARGET_CLASSES)]
        if unclassified.empty:
            logger.info("There are no unclassified objects to predict")
            return {"objects": 0, "explained": 0, "by_type": {}}

        predictor = SatellitesPredictor.load(self.config.output_dir)
        result = predictor.predict_batch(unclassified)
        result["EXPLANATION"] = None

        top = result.sort_values("PROBABILITY", ascending=False)
        for index in top.index[: self.config.explain_top]:
            record = unclassified.loc[index].to_dict()
            prediction = predictor.predict_one(record)
            result.loc[index, "EXPLANATION"] = explain_prediction(
                self.client, record, prediction
            )

        columns = [
            "NORAD_CAT_ID", "SATNAME", "OBJECT_TYPE",
            "PREDICTED_TYPE", "PROBABILITY", "EXPLANATION",
        ]
        result[columns].to_csv(self.config.output_dir / self.PREDICTIONS_FILE, index=False)
        return {
            "objects": len(result),
            "explained": int(result["EXPLANATION"].notna().sum()),
            "by_type": {
                str(label): int(count)
                for label, count in result["PREDICTED_TYPE"].value_counts().items()
            },
        }

    def _fail(self, metrics: dict, status: str, message: str) -> None:
        logger.error(message)
        metrics["status"] = status
        self._write_metrics(metrics)
        raise GateError(message)

    def _write_metrics(self, metrics: dict) -> None:
        path = self.config.output_dir / self.METRICS_FILE
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")