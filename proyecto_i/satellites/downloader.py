import shutil
import kagglehub
from pathlib import Path

dataset_path = Path(__file__).resolve().parent.parent / "data/satellites"

# Download latest version
path = kagglehub.dataset_download(
    "sujaykapadnis/every-known-satellite-orbiting-earth",
    output_dir=dataset_path,
    force_download=True
    )

print("Path to dataset files:", path)

# Remove extra data
extra_csv = Path(dataset_path / "UCS-Satellite-Database-1-1-2023.csv")
extra_csv.unlink()
shutil.rmtree(dataset_path / ".complete")