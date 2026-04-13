from unittest.mock import patch
from application.human_in_loop.feedback_service import handle_user_confirmation


def test_tc_c5_user_confirms_nfr_types():

    # -------------------------
    # Fake input from user (low confidence → confirmed)
    # -------------------------
    items = [
        {
            "title": "Secure Login",
            "description": "System must protect user data",
            "type": "SECURITY"
        },
        {
            "title": "Fast Response",
            "description": "System should respond quickly",
            "type": "PERFORMANCE"
        }
    ]

    # -------------------------
    # Mock repository + mapping
    # -------------------------
    fake_existing_map = {}

    with patch("application.human_in_loop.feedback_service.load_confirmed_types", return_value=fake_existing_map), \
         patch("application.human_in_loop.feedback_service.save_confirmed_types") as mock_save, \
         patch("application.human_in_loop.feedback_service.get_nfr_label", return_value="MockLabel"):

        # -------------------------
        # Call function
        # -------------------------
        result = handle_user_confirmation(items)

        # -------------------------
        # Assertions
        # -------------------------
        assert result == 2 # 2 items confirmed

        # ensure save was called
        mock_save.assert_called_once()

        # check saved data
        saved_data = mock_save.call_args[0][0]

        assert saved_data["System must protect user data"] == "SECURITY"
        assert saved_data["System should respond quickly"] == "PERFORMANCE"