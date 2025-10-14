import os
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from utils import list_images, save_meta

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
print("Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

yolo = YOLO("yolov8n.pt")  # smallest YOLO for CPU

os.makedirs("data", exist_ok=True)

def compute_image_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = clip_proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs).cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0]

def caption_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = blip_proc(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def detect_objects(img_path):
    results = yolo.predict(img_path, verbose=False)
    names = [yolo.model.names[int(box.cls)] for box in results[0].boxes]
    return list(set(names))[:5]

def main():
    img_paths = list_images()
    all_embs = []
    meta = []
    print(f"Processing {len(img_paths)} images...")
    for idx, path in enumerate(tqdm(img_paths)):
        try:
            emb = compute_image_embedding(path)
            caption = caption_image(path)
            objects = detect_objects(path)
            all_embs.append(emb)
            meta.append({
                "id": idx,
                "filename": os.path.basename(path),
                "path": path,
                "caption": caption,
                "objects": objects
            })
        except Exception as e:
            print(f"Error on {path}: {e}")
    # Save embeddings and metadata
    all_embs = np.vstack(all_embs).astype("float32")
    np.save("data/embeddings.npy", all_embs)
    save_meta(meta)
    # Build FAISS index
    index = faiss.IndexFlatIP(all_embs.shape[1])
    index.add(all_embs)
    faiss.write_index(index, "data/faiss.index")
    print("✅ Embeddings and index built successfully.")

if __name__ == "__main__":
    main()
