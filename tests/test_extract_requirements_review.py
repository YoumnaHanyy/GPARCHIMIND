from unittest.mock import patch
from service.srs_extractor import SRSExtractor


# =========================================
# TC-C6: User reviews extracted requirements (Positive)
# =========================================
def test_tc_c6_review_extracted_requirements():

    fake_output = '''
    {
      "functional": [
        {
          "title": "Login",
          "description": "User logs in"
        }
      ],
      "non_functional": [
        {
          "title": "Security",
          "description": "System must be secure"
        }
      ]
    }
    '''

    with patch("service.srs_extractor.generate", return_value=fake_output):

        extractor = SRSExtractor("fake_key")

        result = extractor.extract_requirements("User logs in")

        # =========================
        # 🔥 STRONG ASSERTIONS
        # =========================

        # existence
        assert "functional" in result
        assert "non_functional" in result

        # not empty
        assert len(result["functional"]) > 0
        assert len(result["non_functional"]) > 0

        # functional check
        assert result["functional"][0]["title"] == "Login"
        assert result["functional"][0]["description"] == "User logs in"

        # non-functional check
        assert result["non_functional"][0]["title"] == "Security"
        assert result["non_functional"][0]["description"] == "System must be secure"