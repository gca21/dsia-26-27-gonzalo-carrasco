from __future__ import annotations

import pytest
import requests

from satellites.downloader import DataSourceError, SpaceTrackDownloader

SATCAT = (
    '"INTLDES","NORAD_CAT_ID","SATNAME"\n'
    '"1957-001A","1","SL-1 R/B"\n'
    '"1957-001B","2","SPUTNIK 1"\n'
)


class FakeResponse:
    def __init__(self, text="", error=None):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


def fake_session(monkeypatch, login=None, query=None):
    """Replaces requests.Session, so no test touches the network"""

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, data, timeout):
            return login or FakeResponse()

        def get(self, url, timeout):
            return query or FakeResponse(SATCAT)

    monkeypatch.setattr("satellites.downloader.requests.Session", Session)


def test_download_writes_the_csv_and_returns_the_rows(tmp_path, monkeypatch):
    # Arrange
    fake_session(monkeypatch)
    path = tmp_path / "data" / "satellites.csv"

    # Act
    rows = SpaceTrackDownloader("user", "password").download(path)

    # Assert
    assert rows == 2
    assert path.read_text(encoding="utf-8") == SATCAT
    assert not (tmp_path / "data" / "satellites.csv.tmp").exists()


def test_download_raises_if_the_request_fails(tmp_path, monkeypatch):
    # Arrange
    fake_session(monkeypatch, login=FakeResponse(error=requests.HTTPError("401 Unauthorized")))

    # Act and assert
    with pytest.raises(DataSourceError, match="request failed"):
        SpaceTrackDownloader("user", "password").download(tmp_path / "satellites.csv")


def test_download_keeps_the_previous_file_if_the_response_is_not_a_csv(tmp_path, monkeypatch):
    # Arrange
    fake_session(monkeypatch, query=FakeResponse('{"Login": "Failed"}'))
    path = tmp_path / "satellites.csv"
    path.write_text("previous data", encoding="utf-8")

    # Act and assert
    with pytest.raises(DataSourceError, match="Unexpected response"):
        SpaceTrackDownloader("user", "password").download(path)
    assert path.read_text(encoding="utf-8") == "previous data"


@pytest.mark.parametrize("user, password", [("", "password"), ("user", None), (None, None)])
def test_downloader_requires_credentials(user, password):
    # Act and assert
    with pytest.raises(DataSourceError, match="credentials"):
        SpaceTrackDownloader(user, password)