from ai.methods.ordinal_method import run_ordinal_method
import pytest


def test_run_ordinal_method_returns_ranked_architectures_from_real_matching():
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


def test_run_ordinal_method_returns_empty_list_when_no_matches():
    predictions = [
        {"predicted_type": "SE", "predicted_level": "High"},
    ]

    architecture_data = [
        {"Architecture": "mvc", "Type": "US", "Level": "Low"},
    ]

    result = run_ordinal_method(predictions, architecture_data)

    assert result == []


def test_run_ordinal_method_raises_error_when_predictions_are_empty():
    with pytest.raises(ValueError, match="No NFR predictions found for Ordinal Method"):
        run_ordinal_method([], [
            {"Architecture": "microservices", "Type": "SE", "Level": "High"}
        ])


def test_run_ordinal_method_raises_error_when_prediction_columns_missing():
    predictions = [
        {"wrong_type": "SE", "wrong_level": "High"}
    ]

    architecture_data = [
        {"Architecture": "microservices", "Type": "SE", "Level": "High"}
    ]

    with pytest.raises(KeyError):
        run_ordinal_method(predictions, architecture_data)


def test_run_ordinal_method_raises_error_when_architecture_columns_missing():
    predictions = [
        {"predicted_type": "SE", "predicted_level": "High"}
    ]

    architecture_data = [
        {"Arch": "microservices", "Type": "SE"}
    ]

    with pytest.raises(KeyError):
        run_ordinal_method(predictions, architecture_data)