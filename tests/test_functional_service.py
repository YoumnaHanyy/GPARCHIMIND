from service.functional_service import execute_functional_method


def test_execute_functional_method_returns_error_when_no_functional_requirements(mocker):
    mocker.patch(
        "service.functional_service.ExtractionRepository.get_functional",
        return_value=[]
    )

    result = execute_functional_method(project_id=1)

    assert result == {
        "top_architectures": [],
        "error": "No functional requirements found"
    }


def test_execute_functional_method_calls_real_flow_dependencies(mocker):
    fake_reqs = [
        {"title": "User Login", "description": "The user can login and access the system."},
        {"title": "External API", "description": "The system integrates with an external api."}
    ]

    fake_result = {
        "top_architectures": [
            {"Architecture": "client_server", "Score": 1.0, "Reason": "test reason"}
        ],
        "signals": {
            "user_driven": 1,
            "event_triggered": 0,
            "workflow_oriented": 0,
            "extensible": 0,
            "integration_heavy": 1,
            "data_centric": 0
        }
    }

    mocker.patch(
        "service.functional_service.ExtractionRepository.get_functional",
        return_value=fake_reqs
    )
    mock_choose = mocker.patch(
        "service.functional_service.choose_architecture",
        return_value=fake_result
    )
    mock_save = mocker.patch("service.functional_service.save_functional_method")

    result = execute_functional_method(project_id=10)

    assert result == fake_result
    mock_choose.assert_called_once_with(fake_reqs)
    mock_save.assert_called_once_with(10, fake_result)