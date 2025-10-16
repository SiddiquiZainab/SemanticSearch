import os
import json
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def load_meta(path="data/meta.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_meta(meta, path="data/meta.json"):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def list_images(folder="data/images", exts=(".jpg", ".png", ".jpeg")):
    # Convert to absolute path based on project root
    folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), folder)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def open_image(path):
    return Image.open(path).convert("RGB")

def generate_explanation_with_gemini(query, caption, objects, score):
    prompt = f"""
    The user searched for: "{query}".
    The image contains the following objects: {objects}.
    The image caption is: "{caption}".
    The similarity score between the image and the query is {score:.2f}.

    Write one short natural explanation (2 sentences max) about why this image is a relevant match for the query.
    Avoid repeating the score and do not mention confidence values.
    """
    try:
        response = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Automatically matched based on visual and semantic relevance."
