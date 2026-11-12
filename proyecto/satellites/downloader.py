from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """Error getting the data from the source (Space-Track)."""


class DataDownloader(Protocol):
    """Interface for downloading data"""

    def download(self, path: Path) -> int: ...


class SpaceTrackDownloader:
    """Downloads the current SATCAT from Space-Track as csv"""

    BASE = "https://www.space-track.org"
    QUERY = "/basicspacedata/query/class/satcat/CURRENT/Y/orderby/NORAD_CAT_ID asc/format/csv"
    LOGIN_TIMEOUT = 60
    QUERY_TIMEOUT = 300

    def __init__(self, user: str | None, password: str | None) -> None:
        if not user or not password:
            raise DataSourceError(
                "Space-Track credentials are required (SPACETRACK_USER, SPACETRACK_PASS)"
            )
        self._credentials = {"identity": user, "password": password}

    def download(self, path: Path) -> int:
        """Replaces the csv in path with the current SATCAT. Returns the rows downloaded"""

        try:
            with requests.Session() as session:
                login = session.post(
                    f"{self.BASE}/ajaxauth/login",
                    data=self._credentials,
                    timeout=self.LOGIN_TIMEOUT,
                )
                login.raise_for_status()
                response = session.get(self.BASE + self.QUERY, timeout=self.QUERY_TIMEOUT)
                response.raise_for_status()
        except requests.RequestException as error:
            raise DataSourceError(f"Space-Track request failed: {error}") from error

        lines = response.text.splitlines()
        if not lines or "NORAD_CAT_ID" not in lines[0]:
            # A failed login may not return an HTTP error, so the content is checked
            raise DataSourceError(
                f"Unexpected response from Space-Track (login failed?): {response.text[:100]!r}"
            )

        # A failed download never corrupts the csv
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(response.text, encoding="utf-8")
        temporary.replace(path)

        rows = len(lines) - 1
        logger.info("Downloaded %d rows from Space-Track to %s", rows, path)
        return rows