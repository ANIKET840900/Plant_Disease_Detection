from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_reports_setup_without_fake_models() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models_loaded"] is False
    assert payload["inference_ready"] is payload["classifier_loaded"]


def test_classes_are_not_fabricated_before_dataset_setup() -> None:
    response = client.get("/classes")

    assert response.status_code == 200
    assert response.json()["classes"] == []


def test_predict_enforces_upload_size_limit(monkeypatch) -> None:
    from backend.app.config import settings

    monkeypatch.setattr(settings, "max_upload_size_bytes", 4)
    response = client.post("/predict", files={"file": ("sample.jpg", b"12345", "image/jpeg")})

    assert response.status_code == 413
