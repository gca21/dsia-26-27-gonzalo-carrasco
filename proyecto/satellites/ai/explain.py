from __future__ import annotations

import logging

import pandas as pd

from satellites.ai.client import AIClientError, ModelClient

logger = logging.getLogger(__name__)

ANSWER_LANGUAGE = "Spanish"

# Columns of the object that are useful to justify the prediction
ORBIT_FIELDS = [
    "PERIOD", "INCLINATION", "APOGEE", "PERIGEE", "ORBIT_CLASS",
    "LAUNCH_YEAR", "COUNTRY", "RCS_SIZE", "DECAY",
]


def build_prompt(record: dict, prediction: dict) -> str:
    """One single user message, so it works with models without system role"""

    facts = "\n".join(
        f"- {field}: {record[field]}"
        for field in ORBIT_FIELDS
        if field in record and not pd.isna(record[field])
    )
    probabilities = ", ".join(
        f"{label}: {value:.0%}" for label, value in prediction["probabilities"].items()
    )
    return (
        "A machine learning model classified a space object from the satellite "
        "catalog. Explain the result to a non-expert in 2 or 3 sentences.\n"
        "Use ONLY the data below. Do not invent values and do not change the "
        "prediction.\n\n"
        f"Predicted type: {prediction['object_type']}\n"
        f"Model probabilities: {probabilities}\n"
        f"Object data (units: minutes, degrees, km):\n{facts}\n\n"
        f"Answer in {ANSWER_LANGUAGE}."
    )


def explain_prediction(client: ModelClient, record: dict, prediction: dict) -> str | None:
    """Returns the explanation, or None if the AI provider fails.

    The explanation is optional: a provider failure must not break the pipeline.
    """
    try:
        text = client.complete(build_prompt(record, prediction)).strip()
    except AIClientError as error:
        logger.warning("Explanation skipped: %s", error)
        return None
    if not text:
        logger.warning("Explanation skipped: empty response from %s", client.name)
        return None
    return text