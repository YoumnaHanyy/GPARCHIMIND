import importlib
import sys
import types

from ai.methods.functional_method import choose_architecture
from ai.methods.ordinal_method import run_ordinal_method
from service.weighted_service import execute_weighted_method
from service.hybrid_service import execute_hybrid_method


# ============================================================
# Helpers for TC-E1 (Binary)
# ============================================================

class FakeTokenizer:
    def __call__(self, sentence, return_tensors=None, truncation=None, padding=None, max_length=None):
        return {"input_ids": [1, 2, 3]}


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **tokens):
        import torch
        return FakeModelOutput(torch.tensor([[0.1, 0.9]]))  # always predicts 1


class FakeCollection:
    def __init__(self, data):
        self.data = data

    def find(self, *args, **kwargs):
        return self.data


class FakeDB:
    def __init__(self):
        self.collections = {
            "nfr_predictions": FakeCollection([
                {"description": "System must be secure", "predicted_type": "SE"},
                {"description": "System must be scalable", "predicted_type": "SC"},
            ]),
            "ArchitectureDataset": FakeCollection([
                {"Architecture": "Microservices", "Type": "SE", "label": 1},
                {"Architecture": "Microservices", "Type": "SC", "label": 1},
                {"Architecture": "Layered", "Type": "SE", "label": 1},
                {"Architecture": "Layered", "Type": "SC", "label": 0},
            ]),
        }

    def __getitem__(self, name):
        return self.collections[name]


# ============================================================
# TC-E1
# Verify binary classification of NFRs and ranked architectures
# ============================================================

def test_TC_E1_binary_architecture_recommendation(monkeypatch):
    fake_transformers = types.SimpleNamespace()

    class FakeBertTokenizer:
        @staticmethod
        def from_pretrained(path):
            return FakeTokenizer()

    class FakeBertModel:
        @staticmethod
        def from_pretrained(path):
            return FakeModel()

    fake_transformers.BertTokenizer = FakeBertTokenizer
    fake_transformers.BertForSequenceClassification = FakeBertModel

    fake_database_module = types.SimpleNamespace(db=FakeDB())

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "infrastructure.database", fake_database_module)

    if "ai.inference.predict_binary" in sys.modules:
        del sys.modules["ai.inference.predict_binary"]

    predict_binary_module = importlib.import_module("ai.inference.predict_binary")
    result = predict_binary_module.predict_binary_architecture()

    assert "srs_vector" in result
    assert "top_architectures" in result
    assert len(result["top_architectures"]) > 0
    assert result["top_architectures"][0]["architecture"] == "Microservices"


# ============================================================
# TC-E2
# Verify FR-based architecture score calculation
# ============================================================

def test_TC_E2_functional_architecture_scoring():
    functional_reqs = [
        {
            "title": "User Login",
            "description": "The user can login and access the system based on role."
        },
        {
            "title": "External Payment API",
            "description": "The system integrates with an external third-party api."
        },
        {
            "title": "Order Workflow",
            "description": "The process follows multiple workflow steps."
        }
    ]

    result = choose_architecture(functional_reqs)

    assert "top_architectures" in result
    assert "signals" in result
    assert len(result["top_architectures"]) > 0
    assert result["signals"]["user_driven"] > 0
    assert result["signals"]["integration_heavy"] > 0
    assert result["signals"]["workflow_oriented"] > 0
    assert result["top_architectures"][0]["Architecture"] == "client_server"


# ============================================================
# TC-E3
# Verify ordinal classification matching with architectures
# ============================================================

def test_TC_E3_ordinal_architecture_matching():
    predictions = [
        {"predicted_type": "SE", "predicted_level": "High"},
        {"predicted_type": "SC", "predicted_level": "Medium"},
        {"predicted_type": "SE", "predicted_level": "High"},
    ]

    architecture_data = [
        {"Architecture": "microservices", "Type": "SE", "Level": "High"},
        {"Architecture": "microservices", "Type": "SC", "Level": "Medium"},
        {"Architecture": "layered", "Type": "SE", "Level": "High"},
        {"Architecture": "mvc", "Type": "US", "Level": "Low"},
    ]

    result = run_ordinal_method(predictions, architecture_data)

    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["architecture"] == "microservices"
    assert result[0]["matched_nfrs"] == 3
    assert result[1]["architecture"] == "layered"
    assert result[1]["matched_nfrs"] == 2


# ============================================================
# TC-E4
# Verify weighted NFR computation and normalized architecture scores
# ============================================================

def test_TC_E4_weighted_architecture_scoring(mocker):
    freq_norm = {"SE": 1.0, "SC": 0.5}
    must_norm = {"SE": 1.0, "SC": 1.0}
    importance = {"SE": 0.6, "SC": 0.4}

    fake_arch_data = [
        {"Architecture": "Microservices", "Type": "SE", "LevelNorm": 0.9},
        {"Architecture": "Microservices", "Type": "SC", "LevelNorm": 0.8},
        {"Architecture": "Layered", "Type": "SE", "LevelNorm": 0.7},
        {"Architecture": "Layered", "Type": "SC", "LevelNorm": 0.4},
    ]

    mocker.patch(
        "service.weighted_service.get_architecture_dataset",
        return_value=fake_arch_data
    )

    result = execute_weighted_method(freq_norm, must_norm, importance)

    assert "top_architectures" in result
    assert len(result["top_architectures"]) > 0
    assert result["top_architectures"][0]["Architecture"] == "Microservices"
    assert result["top_architectures"][0]["Score"] >= result["top_architectures"][1]["Score"]


# ============================================================
# TC-E5
# Verify hybrid service ranks architectures and saves final recommendation
# ============================================================

def test_TC_E5_hybrid_architecture_recommendation(mocker):
    functional = {
        "top_architectures": [
            {"Architecture": "Microservices", "Score": 1.0},
            {"Architecture": "Layered", "Score": 0.6},
        ]
    }

    ordinal = {
        "result": [
            {"architecture": "Microservices", "matched_nfrs": 3},
            {"architecture": "Layered", "matched_nfrs": 1},
        ]
    }

    binary = {
        "top_5_architectures": [
            {"architecture": "Microservices", "score": 1.0},
            {"architecture": "Layered", "score": 0.5},
        ]
    }

    weighted = {
        "top_architectures": [
            {"Architecture": "Microservices", "Score": 0.95},
            {"Architecture": "Layered", "Score": 0.55},
        ]
    }

    fake_result = [
        {"Architecture": "Microservices", "FinalScore": 100.0},
        {"Architecture": "Layered", "FinalScore": 63.16},
    ]

    mock_hybrid = mocker.patch(
        "service.hybrid_service.hybrid_aggregation",
        return_value=fake_result
    )

    mock_save = mocker.patch("service.hybrid_service.save_hybrid_result")

    result = execute_hybrid_method(
        project_id=101,
        functional=functional,
        ordinal=ordinal,
        binary=binary,
        weighted=weighted
    )

    assert result == fake_result

    mock_hybrid.assert_called_once_with(
        functional=functional,
        ordinal=[
            {"Architecture": "Microservices", "MatchedNFRs": 3},
            {"Architecture": "Layered", "MatchedNFRs": 1},
        ],
        binary=[
            {"architecture": "Microservices", "score": 1.0},
            {"architecture": "Layered", "score": 0.5},
        ],
        weighted=weighted
    )

    mock_save.assert_called_once_with(
        project_id=101,
        result=fake_result,
        selected_architecture="Unknown"
    )