from unittest.mock import patch, MagicMock
from service.srs_extractor import SRSExtractor
from presentation.routes.srs_routes import validate_pdf_file
from io import BytesIO
from fastapi import UploadFile


def test_tc_c1_valid_pdf_upload():

    # simulate valid PDF file
    file = UploadFile(
        filename="test.pdf",
        file=BytesIO(b"%PDF-1.4 fake pdf content")
    )

    result = validate_pdf_file(file)

    # =========================
    # assertions
    # =========================
    assert result is None  # means file is valid






def test_tc_d1_invalid_file_upload():

  
    # simulate invalid file (not PDF)
    file = UploadFile(
        filename="test.txt",
        file=BytesIO(b"not a pdf")
    )

    result = validate_pdf_file(file)

    # =========================
    # assertions
    # =========================
    assert result is not None
    assert "Invalid file format" in result



    

def test_tc_c2_extract_functional_requirements():

    # -------------------------
    # fake client
    # -------------------------
    fake_client = MagicMock()

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message={
            "content": '''
            {
              "functional": [
                {
                  "title": "Login",
                  "description": "User shall login"
                }
              ],
              "non_functional": []
            }
            '''
        })
    ]

    fake_client.chat.completions.create.return_value = fake_response

    # -------------------------
    # patch client بالكامل
    # -------------------------
    with patch("service.srs_extractor.generate", return_value=fake_response.choices[0].message["content"]), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_functional"), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_non_functional"), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_extraction_results", return_value={"file": "path.json"}):

        extractor = SRSExtractor("fake_key")

        result = extractor.extract_requirements("User shall login")

        # -------------------------
        # assertions
        # -------------------------
        assert len(result["functional"]) == 1
        assert result["functional"][0]["description"]["description"] == "User shall login"











def test_tc_c3_extract_nfr():
    
    fake_client = MagicMock()

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message={
            "content": '''
            {
              "functional": [],
              "non_functional": [
                {
                  "title": "Performance",
                  "description": "System shall respond within 2 seconds"
                },
                {
                  "title": "Security",
                  "description": "System shall use encryption"
                }
              ]
            }
            '''
        })
    ]

    fake_client.chat.completions.create.return_value = fake_response

    with patch("service.srs_extractor.generate", return_value=fake_response.choices[0].message["content"]), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_functional"), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_non_functional"), \
         patch("infrastructure.repositories.extraction_repository.ExtractionRepository.save_extraction_results", return_value={"file": "path.json"}):

        extractor = SRSExtractor("fake_key")

        result = extractor.extract_requirements("System shall be fast and secure")

        # -------------------------
        # assertions
        # -------------------------
        assert "non_functional" in result
        assert len(result["non_functional"]) == 2

        assert result["non_functional"][0]["title"] == "Performance"
        assert result["non_functional"][1]["title"] == "Security"











