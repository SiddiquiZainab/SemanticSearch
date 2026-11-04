import json
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_data_dir() -> Path:
    """Resolve the external data directory independently of the working directory."""
    configured = os.getenv("SEMANTIC_SEARCH_DATA_DIR", "data").strip() or "data"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_DIR = get_data_dir()


def load_meta(path=None):
    metadata_path = Path(path) if path is not None else DATA_DIR / "meta.json"
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    if not metadata_path.exists():
        return []
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_meta(meta, path=None):
    metadata_path = Path(path) if path is not None else DATA_DIR / "meta.json"
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def list_images(folder=None, exts=(".jpg", ".jpeg", ".png")):
    image_dir = Path(folder) if folder is not None else DATA_DIR / "images"
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir
    image_dir = image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. "
            "Set SEMANTIC_SEARCH_DATA_DIR and provide an images/ directory."
        )
    return sorted(
        str(path) for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in exts
    )


def open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def generate_explanation_with_gemini(query, caption, objects, score):
    """Gemini is optional; retrieval remains available without an API key."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.lower().startswith("replace_with"):
        return "Matched using visual and semantic similarity."
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        prompt = f"""
The user searched for: "{query}".
The image objects are: {objects}.
The image caption is: "{caption}".
The image-query similarity score is {score:.2f}.

Write one short explanation (maximum two sentences) describing why the image
may be relevant. Do not repeat the score or claim certainty.
"""
        response = genai.GenerativeModel(
            "gemini-2.0-flash-lite"
        ).generate_content(prompt)
        return (response.text or "").strip() or (
            "Matched using visual and semantic similarity."
        )
    except Exception:
        return "Matched using visual and semantic similarity."
