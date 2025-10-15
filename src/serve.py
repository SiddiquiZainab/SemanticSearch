from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch, faiss, numpy as np
from transformers import CLIPProcessor, CLIPModel
from src.utils import load_meta, build_explanation
import os

app = FastAPI(title="Semantic Image Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.read_index("data/faiss.index")
meta = load_meta("data/meta.json")

@app.get("/health")
def health():
    return {"status": "ok", "count": len(meta)}

@app.post("/search")
async def search(payload: dict):
    query = payload.get("query", "")
    inputs = clip_proc(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_emb = clip_model.get_text_features(**inputs).cpu().numpy()
    txt_emb = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
    D, I = index.search(txt_emb.astype("float32"), 5)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = meta[idx]
        explanation = build_explanation(query, m, score)
        img_path = m["path"]
        results.append({
            "filename": os.path.basename(img_path),
            "caption": m["caption"],
            "objects": m["objects"],
            "score": float(score),
            "explanation": explanation
        })
    return {"query": query, "results": results}
