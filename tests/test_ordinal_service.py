from service.ordinal_service import execute_ordinal_method


def test_execute_ordinal_method_loads_predictions_runs_logic_and_saves(mocker):
    fake_predictions = [
        {"predicted_type": "SE", "predicted_level": "High"},
        {"predicted_type": "SC", "predicted_level": "Medium"},
    ]

    fake_architecture_data = [
        {"Architecture": "microservices", "Type": "SE", "Level": "High"},
        {"Architecture": "microservices", "Type": "SC", "Level": "Medium"},
        {"Architecture": "layered", "Type": "SE", "Level": "High"},
    ]

    fake_result = [
        {"architecture": "microservices", "matched_nfrs": 2},
        {"architecture": "layered", "matched_nfrs": 1},
    ]

    mock_file = mocker.mock_open(read_data="dummy")
    mocker.patch("builtins.open", mock_file)
    mocker.patch("service.ordinal_service.json.load", return_value=fake_predictions)

    mock_get_arch = mocker.patch(
        "service.ordinal_service.get_architecture_dataset",
        return_value=fake_architecture_data
    )

    mock_run = mocker.patch(
        "service.ordinal_service.run_ordinal_method",
        return_value=fake_result
    )

    mock_save = mocker.patch(
        "service.ordinal_service.save_ordinal_result",
        return_value="507f1f77bcf86cd799439011"
    )

    result = execute_ordinal_method()

    assert result == {
        "id": "507f1f77bcf86cd799439011",
        "result": fake_result
    }

    mock_get_arch.assert_called_once()
    mock_run.assert_called_once_with(fake_predictions, fake_architecture_data)
    mock_save.assert_called_once_with({
        "method": "ordinal",
        "result": fake_result
    })