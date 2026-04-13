from unittest.mock import patch
from service.srs_extractor import SRSExtractor


def test_tc_c6_success():

    fake_output = '''
    {
      "functional": ["User logs in"],
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

        assert len(result["functional"]) == 1
        assert result["functional"][0]["description"] == "User logs in"

        assert len(result["non_functional"]) == 1
        assert result["non_functional"][0]["title"] == "Security"


def test_tc_c6_llm_failure():

    with patch("service.srs_extractor.generate", side_effect=Exception("API failed")):

        extractor = SRSExtractor("fake_key")

        result = extractor.extract_requirements("text")

        assert result["functional"] == []
        assert result["non_functional"] == []
        assert result["error"] == "LLM extraction failed"





def test_tc_c6_empty_output():

    fake_output = '''
    {
      "functional": [],
      "non_functional": []
    }
    '''

    with patch("service.srs_extractor.generate", return_value=fake_output):

        extractor = SRSExtractor("fake_key")

        result = extractor.extract_requirements("text")

        assert result["functional"] == []
        assert result["non_functional"] == []