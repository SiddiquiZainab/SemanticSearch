import os
import json
from PIL import Image

def load_meta(path="data/meta.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_meta(meta, path="data/meta.json"):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def list_images(folder="data/images"):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def open_image(path):
    return Image.open(path).convert("RGB")

def build_explanation(query, meta_entry, score):
    query_tokens = set(query.lower().split())
    objects = meta_entry.get("objects", [])
    matched = [o for o in objects if o.lower() in query_tokens]
    explanation = ""
    if matched:
        explanation += f"The image includes {', '.join(matched)} which match your query. "
    caption = meta_entry.get("caption", "")
    if caption:
        explanation += f"Caption: \"{caption}\". "
    explanation += f"(Similarity score: {score:.2f})"
    return explanation.strip()
