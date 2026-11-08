from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from satellites.features import SatellitesFeatureBuilder

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Error training the model."""


class SatellitesTrainer:
    """Trains and saves a classifier of space object types"""

    MODEL_FILE = "model.joblib"
    METADATA_FILE = "metadata.json"

    TARGET_CLASSES = ["PAYLOAD", "ROCKET BODY", "DEBRIS"]

    N_SPLITS = 5
    RANDOM_STATE = 42
    N_TREES = 100
    MIN_ROWS = 10000

    def __init__(self, builder: SatellitesFeatureBuilder | None = None) -> None:
        self.builder = builder or SatellitesFeatureBuilder()

    def prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Returns (X, y, groups). Groups identify the launch event of each object"""

        df = df.reset_index(drop=True)
        target = self.builder.TARGET
        keep = df[target].isin(self.TARGET_CLASSES)
        if (~keep).any():
            logger.info(
                "Discarded %d rows with other classes: %s",
                (~keep).sum(),
                df.loc[~keep, target].value_counts(dropna=False).to_dict(),
            )
        df = df.loc[keep]
        if len(df) < self.MIN_ROWS:
            raise TrainingError(f"Not enough rows to train: {len(df)} < {self.MIN_ROWS}")

        X, y = self.builder.select_features(df)
        if y.nunique() < 2:
            raise TrainingError("At least two classes are needed to train")

        # Fragments of the same event share launch date and site
        groups = df["LAUNCH"].astype(str).str[:10] + "|" + df["SITE"].astype(str)
        return X, y, groups.loc[X.index]

    def split(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series
    ) -> tuple[list[int], list[int]]:
        """Stratified split by class that never separates objects of the same launch"""

        splitter = StratifiedGroupKFold(
            n_splits=self.N_SPLITS, shuffle=True, random_state=self.RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if set(y.iloc[test_idx]) != set(y.unique()):
            logger.warning("Some classes are missing in the test set")
        return train_idx, test_idx

    def build_model(self) -> Pipeline:
        """Preprocessing + classifier. The preprocessing is fitted only with train data"""

        return Pipeline([
            ("preprocess", self.builder.build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=self.N_TREES,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.RANDOM_STATE,
            )),
        ])

    @staticmethod
    def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
        """Metrics of the model and of a baseline that always predicts the main class"""

        labels = list(model.classes_)
        y_pred = model.predict(X_test)

        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)

        report = classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        return {
            "accuracy": float(report["accuracy"]),
            "f1_macro": float(f1_score(y_test, y_pred, labels=labels, average="macro")),
            "baseline_f1_macro": float(
                f1_score(y_test, baseline_pred, labels=labels, average="macro")
            ),
            "per_class": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                }
                for label in labels
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            },
        }

    def save(self, model: Pipeline, metadata: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / self.MODEL_FILE
        metadata_path = output_dir / self.METADATA_FILE
        joblib.dump(model, model_path)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Model saved in %s and metadata in %s", model_path, metadata_path)

    def train(self, df: pd.DataFrame, output_dir: Path) -> dict:
        """Trains, evaluates and saves the model. Returns the metrics"""

        X, y, groups = self.prepare(df)
        train_idx, test_idx = self.split(X, y, groups)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logger.info("Split: %d train rows, %d test rows", len(X_train), len(X_test))
        logger.info("Train classes: %s", y_train.value_counts().to_dict())
        logger.info("Test classes: %s", y_test.value_counts().to_dict())

        model = self.build_model()
        logger.info(
            "Training RandomForest (%d trees, balanced class weights)", self.N_TREES
        )
        model.fit(X_train, y_train)

        metrics = self.evaluate(model, X_train, y_train, X_test, y_test)
        logger.info(
            "Macro F1: %.3f (baseline %.3f), accuracy: %.3f",
            metrics["f1_macro"], metrics["baseline_f1_macro"], metrics["accuracy"],
        )

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "random_state": self.RANDOM_STATE,
            "split": f"StratifiedGroupKFold(n_splits={self.N_SPLITS}), groups=LAUNCH|SITE",
            "classes": list(model.classes_),
            "features": {
                "numeric": self.builder.NUMERIC,
                "categorical": self.builder.CATEGORICAL,
            },
            "rows": {"total": len(X), "train": len(X_train), "test": len(X_test)},
            "metrics": metrics,
        }
        self.save(model, metadata, output_dir)
        return metrics


if __name__ == "__main__":
    from satellites.loader import CsvSatellitesRepository
    from satellites.validator import SatellitesValidator

    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    root = Path(__file__).resolve().parent.parent
    raw = CsvSatellitesRepository(root / "data/satellites/satellites.csv").load()
    satellites, errors = SatellitesValidator(raw).validate()
    print(f"Valid rows: {len(satellites)}, invalid rows: {len(errors)}")

    metrics = SatellitesTrainer().train(satellites, root / "models")

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['f1_macro']:.3f} (baseline {metrics['baseline_f1_macro']:.3f})")
    for label, values in metrics["per_class"].items():
        print(
            f"{label:12s} precision={values['precision']:.2f} "
            f"recall={values['recall']:.2f} f1={values['f1']:.2f} "
            f"support={values['support']}"
        )