from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    def test_health_check():
        response = client.get("/health")
        assert response.status_code == 200

    def test_predict_invalid_file_type():
        response = client.post("/predict", files={"file": ("test.txt", b"Texte", "text/plain")})
        assert response.status_code == 400
