from ai.methods.functional_method import extract_behavior_signals, choose_architecture


def test_extract_behavior_signals_counts_real_fr_keywords():
    functional_reqs = [
        {
            "title": "User Login",
            "description": "The user can login and access the system based on role."
        },
        {
            "title": "Order Workflow",
            "description": "The system processes each step in the workflow."
        },
        {
            "title": "External Payment API",
            "description": "The system integrates with an external third-party api."
        },
        {
            "title": "Data Storage",
            "description": "The system stores and updates data in the database."
        },
        {
            "title": "Notification Event",
            "description": "The system sends notifications when an event is triggered."
        },
        {
            "title": "Plugin Support",
            "description": "The system allows plugin extension and add new modules."
        }
    ]

    signals = extract_behavior_signals(functional_reqs)

    assert signals["user_driven"] > 0
    assert signals["workflow_oriented"] > 0
    assert signals["integration_heavy"] > 0
    assert signals["data_centric"] > 0
    assert signals["event_triggered"] > 0
    assert signals["extensible"] > 0


def test_choose_architecture_returns_ranked_architectures_based_on_real_fr_signals():
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
            "title": "Notification Event",
            "description": "The system sends notifications when an event is triggered."
        },
        {
            "title": "Data Storage",
            "description": "The system stores and updates data in the database."
        }
    ]

    result = choose_architecture(functional_reqs)

    assert "top_architectures" in result
    assert "signals" in result
    assert isinstance(result["top_architectures"], list)
    assert len(result["top_architectures"]) > 0

    top_archs = result["top_architectures"]
    signals = result["signals"]

    assert signals["user_driven"] > 0
    assert signals["integration_heavy"] > 0
    assert signals["event_triggered"] > 0
    assert signals["data_centric"] > 0

    for item in top_archs:
        assert "Architecture" in item
        assert "Score" in item
        assert "Reason" in item
        assert 0 <= item["Score"] <= 1