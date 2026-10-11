import os
from pathlib import Path

import requests
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

dataset_path = project_root / "data/satellites"
dataset_path.mkdir(parents=True, exist_ok=True)

BASE = "https://www.space-track.org"
QUERY = "/basicspacedata/query/class/satcat/CURRENT/Y/orderby/NORAD_CAT_ID asc/format/csv"

# .env credentials for space-strack
credentials = {
    "identity": os.environ["SPACETRACK_USER"],
    "password": os.environ["SPACETRACK_PASS"],
}

with requests.Session() as session:
    session.post(f"{BASE}/ajaxauth/login", data=credentials, timeout=60).raise_for_status()
    response = session.get(BASE + QUERY, timeout=300)
    response.raise_for_status()

csv_path = dataset_path / "satellites.csv"
csv_path.write_text(response.text, encoding="utf-8")

print("Path to dataset files:", csv_path)