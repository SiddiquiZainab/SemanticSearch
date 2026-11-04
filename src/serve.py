import os
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import CLIPModel, CLIPProcessor

from src.utils import DATA_DIR, generate_explanation_with_gemini, load_meta

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Semantic Image Search API",
    description="Natural-language image retrieval using CLIP embeddings and FAISS.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


def load_search_resources():
    index_path = DATA_DIR / "faiss.index"
    metadata_path = DATA_DIR / "meta.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"Search artifacts are missing from {DATA_DIR}. "
            "Run python -m src.compute_embeddings after preparing the dataset."
        )

    metadata = load_meta(metadata_path)
    if not metadata:
        raise RuntimeError(f"No image metadata found in {metadata_path}.")
    vector_index = faiss.read_index(str(index_path))
    if vector_index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata are out of sync: "
            f"{vector_index.ntotal} vectors versus {len(metadata)} metadata records."
        )

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor, vector_index, metadata


clip_model, clip_processor, index, meta = load_search_resources()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_images": len(meta),
        "index_vectors": index.ntotal,
    }


@app.post("/search")
async def search(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    inputs = clip_processor(
        text=[query], return_tensors="pt", padding=True
    ).to(DEVICE)
    with torch.inference_mode():
        text_embedding = clip_model.get_text_features(**inputs).cpu().numpy()

    norm = np.linalg.norm(text_embedding, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise HTTPException(
            status_code=500,
            detail="The query encoder returned an invalid embedding.",
        )
    text_embedding = (text_embedding / norm).astype("float32")
    scores, indices = index.search(text_embedding, TOP_K)
    results = []

    for score, item_index in zip(scores[0], indices[0]):
        if item_index < 0 or score < SIMILARITY_THRESHOLD:
            continue
        record = meta[int(item_index)]
        explanation = generate_explanation_with_gemini(
            query,
            record.get("caption", ""),
            record.get("objects", []),
            float(score),
        )
        results.append({
            "filename": record.get("filename") or Path(record.get("path", "")).name,
            "caption": record.get("caption", ""),
            "objects": record.get("objects", []),
            "score": float(score),
            "explanation": explanation,
        })

    return {
        "query": query,
        "results": results,
        "message": None if results else "No images match your query.",
    }
