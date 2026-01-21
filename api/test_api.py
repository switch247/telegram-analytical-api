import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_top_products():
    response = client.get("/api/reports/top-products?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 3
    if data:
        assert "product" in data[0]
        assert "count" in data[0]
