from __future__ import annotations

from types import SimpleNamespace

import pytest

from satellites.ai import client as client_module
from satellites.ai.client import AIClientError, HFClient, MockClient, get_client


class FakeInferenceClient:
    """Replaces huggingface_hub.InferenceClient, so no test touches the network"""

    def __init__(self, model, provider, token, timeout):
        self.model = model

    def chat_completion(self, messages, max_tokens, temperature):
        content = f"answer from {self.model}: {messages[0]['content']}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FailingInferenceClient(FakeInferenceClient):
    def chat_completion(self, messages, max_tokens, temperature):
        raise RuntimeError("402 Payment Required")


def test_mock_client_is_deterministic():
    # Arrange
    client = MockClient()

    # Act and assert
    assert client.complete("prompt A") == client.complete("prompt B")


def test_get_client_returns_mock_by_default():
    # Act and assert
    assert isinstance(get_client(), MockClient)


@pytest.mark.parametrize(
    "provider, model, token",
    [("hf", "owner/name", None), ("hf", None, "hf_token"), ("other", None, None)],
)
def test_get_client_raises_if_config_is_invalid(provider, model, token):
    # Act and assert
    with pytest.raises(AIClientError):
        get_client(provider, model, token)


def test_hf_client_returns_the_model_answer(monkeypatch):
    # Arrange
    monkeypatch.setattr(client_module, "InferenceClient", FakeInferenceClient)
    client = HFClient(model="owner/name", token="hf_token")

    # Act
    answer = client.complete("hello")

    # Assert
    assert answer == "answer from owner/name: hello"


def test_hf_client_wraps_provider_errors(monkeypatch):
    # Arrange
    monkeypatch.setattr(client_module, "InferenceClient", FailingInferenceClient)
    client = HFClient(model="owner/name", token="hf_token")

    # Act and assert
    with pytest.raises(AIClientError, match="Payment Required"):
        client.complete("hello")


def test_hf_client_passes_the_inference_provider(monkeypatch):
    # Arrange
    captured = {}

    class CapturingClient(FakeInferenceClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(client_module, "InferenceClient", CapturingClient)

    # Act
    get_client("hf", "owner/name", "hf_token", inference_provider="groq")

    # Assert
    assert captured["provider"] == "groq"