import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# -------------------------
# MOCK PREDICTION REPO 🔥
# -------------------------
@pytest.fixture(autouse=True)
def mock_prediction_repo(monkeypatch):

    # 👇 ده أهم mock في حياتك 😂
    monkeypatch.setattr(
        "infrastructure.repositories.nfr_dataset_repository.NFRPredictionRepository.get_by_project",
        lambda project_id: [
            {
                "description": "fast",
                "predicted_type": "PE",
                "predicted_level": "High"
            }
        ]
    )

    # 👇 عشان confirm يعدي
    monkeypatch.setattr(
        "infrastructure.repositories.nfr_dataset_repository.NFRPredictionRepository.confirm_nfr",
        lambda **kwargs: True
    )

    # 👇 عشان level prediction
    monkeypatch.setattr(
        "ai.inference.predict_type_level.predict_level_for_text",
        lambda x: "High"
    )


# -------------------------
# H1
# -------------------------
def test_H1_store():

    payload = {
        "project_id": "test_proj",
        "items": [
            {"description": "fast", "type": "PE"}
        ]
    }

    response = client.post("/confirm_nfr", json=payload)

    assert response.status_code == 200


# -------------------------
# H2
# -------------------------
def test_H2_retrieve():

    payload = {
        "project_id": "test_proj",
        "items": [
            {"description": "fast", "type": "PE"}
        ]
    }

    response = client.post("/confirm_nfr", json=payload)

    data = response.json()

    assert "nfr_predictions" in data
    assert len(data["nfr_predictions"]) > 0


# -------------------------
# H3
# -------------------------
def test_H3_export():

    payload = {
        "project_id": "test_proj",
        "items": [
            {"description": "fast", "type": "PE"}
        ]
    }

    response = client.post("/confirm_nfr", json=payload)

    data = response.json()

    assert "functional_method" in data
    assert "weighted_method" in data
    assert "hybrid_method" in data