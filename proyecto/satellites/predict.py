from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from satellites.features import SatellitesFeatureBuilder
from satellites.train import SatellitesTrainer

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Error loading the model or predicting."""


class SatellitesPredictor:
    """Predicts the type of space objects with a trained model"""

    def __init__(self, model, builder: SatellitesFeatureBuilder | None = None) -> None:
        self.model = model
        self.builder = builder or SatellitesFeatureBuilder()

    @classmethod
    def load(cls, model_dir: Path) -> SatellitesPredictor:
        """Loads the model saved by SatellitesTrainer"""

        model_path = model_dir / SatellitesTrainer.MODEL_FILE
        metadata_path = model_dir / SatellitesTrainer.METADATA_FILE
        if not model_path.exists():
            raise PredictionError(f"Model not found: {model_path}")

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trained_with = metadata.get("sklearn_version")
            if trained_with != sklearn.__version__:
                logger.warning(
                    "Model trained with scikit-learn %s, but %s is installed",
                    trained_with, sklearn.__version__,
                )
        else:
            logger.warning("No metadata found, scikit-learn version not checked")

        logger.info("Model loaded from %s", model_path)
        return cls(joblib.load(model_path))

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of df with the predicted type and its probability"""

        X = self.builder.build_features(df)
        probabilities = self.model.predict_proba(X)

        result = df.copy()
        result["PREDICTED_TYPE"] = self.model.classes_[probabilities.argmax(axis=1)]
        result["PROBABILITY"] = probabilities.max(axis=1)
        logger.info(
            "Predicted %d objects: %s",
            len(result), result["PREDICTED_TYPE"].value_counts().to_dict(),
        )
        return result

    def predict_one(self, record: dict) -> dict:
        """Predicts a single object given as a dict of validated columns"""

        X = self.builder.build_features(pd.DataFrame([record]))
        probabilities = self.model.predict_proba(X)[0]
        best = probabilities.argmax()
        return {
            "object_type": str(self.model.classes_[best]),
            "probability": float(probabilities[best]),
            "probabilities": {
                str(label): float(value)
                for label, value in zip(self.model.classes_, probabilities)
            },
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")

    predictor = SatellitesPredictor.load(root / "models")
    sample = pd.read_csv(root / "data/satellites/clean_satellites.csv").sample(5, random_state=0)
    result = predictor.predict_batch(sample)
    print(result[["SATNAME", "OBJECT_TYPE", "PREDICTED_TYPE", "PROBABILITY"]])