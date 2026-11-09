from __future__ import annotations

import pytest

from satellites.ai.client import AIClientError, MockClient
from satellites.ai.explain import build_prompt, explain_prediction


@pytest.fixture
def record():
    return {
        "PERIOD": 95.2, "INCLINATION": 51.6, "APOGEE": 420.0, "PERIGEE": 410.0,
        "ORBIT_CLASS": "LEO", "LAUNCH_YEAR": 1998, "COUNTRY": "ISS", "RCS_SIZE": "LARGE",
    }


@pytest.fixture
def prediction():
    return {
        "object_type": "PAYLOAD",
        "probability": 0.9,
        "probabilities": {"DEBRIS": 0.05, "PAYLOAD": 0.9, "ROCKET BODY": 0.05},
    }


class FailingClient:
    name = "failing"

    def complete(self, prompt):
        raise AIClientError("provider down")


class EmptyClient:
    name = "empty"

    def complete(self, prompt):
        return "   "


def test_build_prompt_contains_prediction_and_object_data(record, prediction):
    # Act
    prompt = build_prompt(record, prediction)

    # Assert
    assert "Predicted type: PAYLOAD" in prompt
    assert "PAYLOAD: 90%" in prompt
    assert "PERIGEE: 410.0" in prompt


def test_build_prompt_skips_missing_values(record, prediction):
    # Arrange
    record["DECAY"] = float("nan")

    # Act
    prompt = build_prompt(record, prediction)

    # Assert
    assert "DECAY" not in prompt


def test_explain_prediction_returns_the_client_text(record, prediction):
    # Act
    text = explain_prediction(MockClient(), record, prediction)

    # Assert
    assert text.startswith("[mock]")


@pytest.mark.parametrize("client", [FailingClient(), EmptyClient()])
def test_explain_prediction_returns_none_if_client_fails(client, record, prediction):
    # Act and assert
    assert explain_prediction(client, record, prediction) is None