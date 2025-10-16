from fastapi.testclient import TestClient
from src.serve import app

client = TestClient(app)

# ---- Test search endpoint with a normal query ----
def test_search_normal():
    test_query = {"query": "dog"}
    response = client.post("/search", json=test_query)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

# ---- Test search endpoint with empty query ----
def test_search_empty_query():
    test_query = {"query": ""}
    response = client.post("/search", json=test_query)
    assert response.status_code == 200
    data = response.json()
    assert data.get("results") == []
