# Semantic Image Search

A semantic image search engine using **CLIP embeddings**, **FAISS index**, **FastAPI**, **Streamlit**, and **Gemini API** for dynamic explanations.

The app allows users to search images semantically, displaying the top matching images along with captions, detected objects, and explanations.

---

## Features

- Search images using natural language queries.
- Retrieve only highly relevant images using **similarity thresholding**.
- Generate **human-readable explanations** for matches via Gemini API.
- Grid layout display of images with captions, objects, and similarity scores.
- Fully **Dockerized** for easy deployment.

---

## Setup Instructions

### 1. Clone the Repository

```bash

git clone <https://github.com/SiddiquiZainab/SemanticSearch.git>
cd SemanticSearch
```
### 2. Install Requirements
```bash

pip install -r requirements.txt
```

### 3. Add Gemini API Key
Create a .env file in the root folder:
```bash

GEMINI_API_KEY="your_api_key_here"
```

### 4. Compute Image Embeddings
```bash

python src/compute_embeddings.py
```

### 5. Run Locally

FastAPI 
```bash

uvicorn src.serve:app --reload
```
Streamlit

```bash

streamlit run src/app.py
```
### 6. Run with Docker
```bash
docker-compose up --build
```

FastAPI: http://localhost:8000/docs

Streamlit: http://localhost:8501

---

## Usage

1. Enter a search query in Streamlit.

2. View images returned with captions, objects, similarity score, and Gemini explanations.

3. If no matches exceed the threshold, a message will display:
"No images match your query."

### Notes

- Adjust similarity threshold in serve.py to control result relevance.

- Gemini API generates natural-language explanations; ensure GEMINI_API_KEY is valid.

- Dockerized setup ensures reproducible deployments.