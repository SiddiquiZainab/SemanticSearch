import os
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)
from ultralytics import YOLO

from src.utils import DATA_DIR, list_images, save_meta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
BLIP_MODEL_NAME = os.getenv(
    "BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base"
)
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")


def load_models():
    """Load vision models for the offline indexing job."""
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_NAME
    ).to(DEVICE)
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    yolo_model = YOLO(YOLO_MODEL_NAME)
    return clip_model, clip_processor, blip_model, blip_processor, yolo_model


def compute_image_embedding(image_path, model, processor):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        embedding = model.get_image_features(**inputs).cpu().numpy()
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError(f"Zero-norm embedding for {image_path}")
    return (embedding / norm)[0].astype("float32")


def caption_image(image_path, model, processor):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True)


def detect_objects(image_path, model):
    results = model.predict(str(image_path), verbose=False)
    if not results or results[0].boxes is None:
        return []
    names = [model.names[int(box.cls)] for box in results[0].boxes]
    return list(dict.fromkeys(names))[:5]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = list_images()
    if not image_paths:
        raise RuntimeError(
            f"No images found under {DATA_DIR / 'images'}. "
            "Populate the external image directory before indexing."
        )

    clip_model, clip_processor, blip_model, blip_processor, yolo_model = load_models()
    embeddings = []
    metadata = []
    failures = []
    print(f"Indexing {len(image_paths)} images from {DATA_DIR / 'images'}")

    for image_id, image_path in enumerate(tqdm(image_paths)):
        try:
            embedding = compute_image_embedding(
                image_path, clip_model, clip_processor
            )
            caption = caption_image(image_path, blip_model, blip_processor)
            objects = detect_objects(image_path, yolo_model)
            relative_path = Path(image_path).resolve().relative_to(DATA_DIR).as_posix()
            embeddings.append(embedding)
            metadata.append({
                "id": image_id,
                "filename": Path(image_path).name,
                "path": relative_path,
                "caption": caption,
                "objects": objects,
            })
        except Exception as exc:
            failures.append({"path": image_path, "error": str(exc)})
            print(f"Skipping {image_path}: {exc}")

    if not embeddings:
        raise RuntimeError("No images were processed; index was not written.")

    vectors = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    np.save(DATA_DIR / "embeddings.npy", vectors)
    faiss.write_index(index, str(DATA_DIR / "faiss.index"))
    save_meta(metadata, DATA_DIR / "meta.json")
    print(
        f"Index complete: {len(metadata)} indexed; {len(failures)} skipped; "
        f"output: {DATA_DIR}"
    )


if __name__ == "__main__":
    main()
