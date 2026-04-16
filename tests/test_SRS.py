import sys
from unittest.mock import MagicMock

sys.modules["infrastructure.database"] = MagicMock()


import pandas as pd
import torch
from unittest.mock import patch
from io import BytesIO
from fastapi import UploadFile
from service.srs_extractor import SRSExtractor
from application.human_in_loop.feedback_service import handle_user_confirmation
from presentation.routes.srs_routes import validate_pdf_file
from presentation.routes.srs_routes import check_srs_sections
from ai.inference.predict_type_level import predict_and_save_nfr


# =========================================
# TC-C1: Validate file format
# =========================================
def test_tc_c1_valid_pdf_upload():

    # simulate valid PDF file
    file = UploadFile(
        filename="test.pdf",
        file=BytesIO(b"%PDF-1.4 fake pdf content")
    )

    result = validate_pdf_file(file)

    
    assert result is None  # means file is valid

# =========================================
# TC-C2: Extract Functional Requirements
# =========================================
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

# =========================================
# TC-C3: Extract Non Functional Requirements
# =========================================
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




# =========================================
# TC-C4: Correct NFR classification
# =========================================
def test_tc_c4_correct_classification():

    fake_nfrs = [
        {"title": "Performance", "description": "System should respond fast"},
        {"title": "Security", "description": "System must be secure"}
    ]

    # 🔥 fake model بيرجع logits حقيقية
    def fake_model(*args, **kwargs):
        class Model:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return type("obj", (), {
                    "logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]])
                })
        return Model()

    with patch("json.load", return_value=fake_nfrs), \
     patch("builtins.open"), \
     patch(
         "ai.inference.predict_type_level.NFRDatasetRepository.load_nfr_dataset_from_mongo",
         return_value=pd.DataFrame({
             "Type": ["PERFORMANCE", "SECURITY"],
             "Level": ["High", "Low"]  # 🔥 FIX هنا بس
         })
     ), \
     patch("ai.inference.predict_type_level.BertTokenizer.from_pretrained"), \
     patch(
         "ai.inference.predict_type_level.BertForSequenceClassification.from_pretrained",
         side_effect=fake_model
     ), \
     patch("ai.inference.predict_type_level.NFRPredictionRepository.save_batch"):

        # 🔥 نشغل الفنكشن بجد
        result = predict_and_save_nfr("proj_test")

        # =========================
        # 🔥 STRONG ASSERTIONS
        # =========================

        # count
        assert len(result) == 2

        # كل NFR متصنّف
        for item in result:
            assert "predicted_type" in item
            assert item["predicted_type"] is not None
            assert item["predicted_type"] != ""

        # check structure
        assert "confidence" in result[0]
        assert "predicted_level" in result[0]

# =========================================
# TC-C5: Human in Loop
# =========================================
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


# =========================================
# TC-C6: User reviews extracted requirements (Positive)
# =========================================
def test_tc_c6_review_extracted_requirements():

    fake_output = '''
    {
      "functional": [
        {
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
        assert result["functional"][0]["description"]["description"] == "User logs in"

        # non-functional check
        assert result["non_functional"][0]["title"] == "Security"
        assert result["non_functional"][0]["description"] == "System must be secure"

# =========================================  
# TC-D1: Invalid file formate
# =========================================
def test_tc_d1_invalid_file_upload():

  
    # simulate invalid file (not PDF)
    file = UploadFile(
        filename="test.txt",
        file=BytesIO(b"not a pdf")
    )

    result = validate_pdf_file(file)

    assert result is not None
    assert "Invalid file format" in result

# =========================================  
# TC-D2: Empty SRS content
# =========================================
def test_tc_d2_empty_srs():
    text = ""

    has_fr, has_nfr = check_srs_sections(text)

    assert has_fr is False
    assert has_nfr is False


# =========================================
# TC-D2: Random (invalid) SRS content
# =========================================
def test_tc_d2_random_text():
    text = "This is just a random document without requirements"

    has_fr, has_nfr = check_srs_sections(text)

    assert has_fr is False
    assert has_nfr is False