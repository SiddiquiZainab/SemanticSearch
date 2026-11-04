import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from src.utils import DATA_DIR

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("SEARCH_REQUEST_TIMEOUT_SECONDS", "90"))

st.set_page_config(page_title="Semantic Image Search", page_icon="🔎", layout="wide")
st.title("Semantic Image Search")
st.caption("Find visually relevant images using natural-language descriptions.")

query = st.text_input(
    "Search images",
    placeholder="e.g. a dog running through a forest",
    max_chars=500,
)

if st.button("Search", type="primary", disabled=not query.strip()):
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query.strip()},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        st.error("The search request timed out. Please try again.")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Could not complete the search request: {exc}")
        st.stop()
    except ValueError:
        st.error("The search service returned an invalid response.")
        st.stop()

    results = data.get("results", [])
    if not results:
        st.info(data.get("message") or "No images match your query.")
    else:
        st.write(f"Showing {len(results)} matching images.")
        columns_per_row = 3
        for start in range(0, len(results), columns_per_row):
            columns = st.columns(columns_per_row)
            for column, item in zip(columns, results[start:start + columns_per_row]):
                with column:
                    filename = Path(item.get("filename", "")).name
                    image_path = DATA_DIR / "images" / filename
                    if not image_path.is_file():
                        st.warning(f"Image asset is unavailable: {filename}")
                        continue
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    st.image(
                        image,
                        caption=(
                            f"{item.get('caption', 'Image')} "
                            f"(similarity: {item.get('score', 0):.3f})"
                        ),
                        use_container_width=True,
                    )
                    objects = item.get("objects", [])
                    if objects:
                        st.write("Objects: " + ", ".join(objects))
                    if item.get("explanation"):
                        st.write(item["explanation"])
