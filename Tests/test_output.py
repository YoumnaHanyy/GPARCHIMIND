import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# -------------------------
# IMPORT REAL MODULES
# -------------------------
from presentation.routes import srs_routes
from infrastructure.repositories import nfr_dataset_repository
from ai.inference import predict_type_level
from service import nfr_stats_service


# =========================================================
# 🔥 MOCK ALL EXTERNAL DEPENDENCIES
# =========================================================
@pytest.fixture(autouse=True)
def mock_all(monkeypatch):

    # -------------------------
    # DB MOCK
    # -------------------------
    monkeypatch.setattr(
        nfr_dataset_repository.NFRPredictionRepository,
        "get_by_project",
        lambda *args, **kwargs: [
            {
                "description": "fast",
                "predicted_type": "PE",
                "predicted_level": "High"
            }
        ]
    )

    monkeypatch.setattr(
        nfr_dataset_repository.NFRPredictionRepository,
        "confirm_nfr",
        lambda *args, **kwargs: True
    )

    # -------------------------
    # AI MOCK
    # -------------------------
    monkeypatch.setattr(
        predict_type_level,
        "predict_level_for_text",
        lambda *args, **kwargs: "High"
    )

    # -------------------------
    # ARCHITECTURE METHODS MOCK
    # -------------------------
    monkeypatch.setattr(
        srs_routes,
        "execute_functional_method",
        lambda *args, **kwargs: {"functional": "ok"}
    )

    monkeypatch.setattr(
        srs_routes,
        "execute_ordinal_method",
        lambda *args, **kwargs: {"ordinal": "ok"}
    )

    monkeypatch.setattr(
        srs_routes,
        "execute_binary_method",
        lambda *args, **kwargs: {"binary": "ok"}
    )

    monkeypatch.setattr(
        srs_routes,
        "execute_weighted_method",
        lambda *args, **kwargs: {
            "top_architectures": ["Layered"]
        }
    )

    monkeypatch.setattr(
        srs_routes,
        "execute_hybrid_method",
        lambda *args, **kwargs: {"hybrid": "ok"}
    )

    # -------------------------
    # 🔥 PHASE 4 (LLM) MOCK
    # -------------------------
    monkeypatch.setattr(
        srs_routes,
        "generate_phase4",
        lambda *args, **kwargs: {"phase4": "ok"}
    )


# =========================================================
# 🥇 ENDPOINT TESTS
# =========================================================

def test_H1_store():
    payload = {
        "project_id": "test_proj",
        "items": [{"description": "fast", "type": "PE"}]
    }

    response = client.post("/confirm_nfr", json=payload)

    assert response.status_code == 200


def test_H2_retrieve():
    payload = {
        "project_id": "test_proj",
        "items": [{"description": "fast", "type": "PE"}]
    }

    response = client.post("/confirm_nfr", json=payload)
    data = response.json()

    assert "nfr_predictions" in data
    assert len(data["nfr_predictions"]) > 0


def test_H3_export():
    payload = {
        "project_id": "test_proj",
        "items": [{"description": "fast", "type": "PE"}]
    }

    response = client.post("/confirm_nfr", json=payload)
    data = response.json()

    assert "functional_method" in data
    assert "weighted_method" in data
    assert "hybrid_method" in data


# =========================================================
# 🥈 SERVICE TEST
# =========================================================

def test_compute_nfr_statistics():

    sample_data = [
        {"predicted_type": "PE", "predicted_level": "High"},
        {"predicted_type": "PE", "predicted_level": "Medium"},
        {"predicted_type": "SE", "predicted_level": "Low"},
    ]

    freq_norm, must_norm, importance = nfr_stats_service.compute_nfr_statistics(sample_data)

    assert isinstance(freq_norm, dict)
    assert isinstance(must_norm, dict)
    assert isinstance(importance, dict)

    assert "PE" in freq_norm
    assert "SE" in freq_norm

    assert max(freq_norm.values()) <= 1
    assert max(must_norm.values()) <= 1
    assert sum(importance.values()) <= 1.01  # tolerance