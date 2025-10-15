import streamlit as st
import requests
from PIL import Image
import os

# Base URL of your FastAPI server
API_URL = "http://fastapi:8000/search"  #Replace 127.0.0.1 with fastapi for docker

st.set_page_config(page_title="Semantic Image Search", layout="wide")
st.title("Semantic Image Search")

# Input field
query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    response = requests.post(API_URL, json={"query": query})
    data = response.json()

    results = data.get("results", [])

    if not results:
        st.write("No images match your query.")
    else:
        st.write(f"Showing the top {len(results)} results")

        # Display images in a grid layout (3 per row)
        num_columns = 3
        for i in range(0, len(results), num_columns):
            cols = st.columns(num_columns)
            for col, item in zip(cols, results[i:i + num_columns]):
                with col:
                    image_path = os.path.join("data/images", item["filename"])  # Adjust if needed

                    try:
                        img = Image.open(image_path)
                        st.image(img, caption=f"{item['caption']} (Score: {item['score']:.2f})",
                                 use_container_width=True, width=200)
                        st.write(f"Objects: {', '.join(item['objects'])}")
                        st.write(item["explanation"])
                    except FileNotFoundError:
                        st.write(f"Image not found: {image_path}")
