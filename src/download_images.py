from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from src.utils import DATA_DIR


def download_images(num_images=None, csv_path=None, output_dir=None, target_size=(800, 800)):
    """Download and optimize images from an externally supplied URL manifest."""
    manifest_path = Path(csv_path) if csv_path else DATA_DIR / "photos_url.csv"
    image_dir = Path(output_dir) if output_dir else DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"URL manifest not found: {manifest_path}. "
            "Place photos_url.csv in SEMANTIC_SEARCH_DATA_DIR."
        )

    dataframe = pd.read_csv(manifest_path)
    if "photo_image_url" not in dataframe.columns:
        raise ValueError("The URL manifest must contain a photo_image_url column.")
    if num_images is not None:
        dataframe = dataframe.head(num_images)

    print(f"Downloading {len(dataframe)} images to {image_dir}")
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        url = str(row["photo_image_url"]).strip()
        if not url or url.lower() == "nan":
            continue
        output_path = image_dir / f"{index + 1:04d}.jpg"
        if output_path.exists():
            continue
        try:
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as source:
                image = source.convert("RGB")
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                image.save(output_path, "JPEG", quality=85, optimize=True)
        except (requests.RequestException, OSError, ValueError) as exc:
            print(f"Failed to download row {index + 1}: {exc}")


if __name__ == "__main__":
    download_images()
