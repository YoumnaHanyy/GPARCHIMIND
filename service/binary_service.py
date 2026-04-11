from service.importance_service import compute_importance_from_srs_pdf


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePdfReader:
    def __init__(self, _file):
        self.pages = [
            FakePage("The system must be secure. The system must scale."),
            FakePage("Availability is required.")
        ]


def test_compute_importance_from_srs_pdf_returns_normalized_scores(mocker):
    mock_file = mocker.mock_open(read_data=b"fake pdf content")
    mocker.patch("builtins.open", mock_file)

    mocker.patch("service.importance_service.PyPDF2.PdfReader", FakePdfReader)

    mocker.patch(
        "service.importance_service.nltk.sent_tokenize",
        return_value=[
            "The system must be secure.",
            "The system must scale.",
            "Availability is required."
        ]
    )

    def fake_predict(sentence):
        mapping = {
            "The system must be secure.": "SECURITY",
            "The system must scale.": "SCALABILITY",
            "Availability is required.": "AVAILABILITY"
        }
        return mapping.get(sentence, "SECURITY")

    mocker.patch(
        "ai.inference.predict_type_level.predict_nfr_type_only",
        side_effect=fake_predict
    )

    result = compute_importance_from_srs_pdf("fake.pdf")

    assert "SECURITY" in result
    assert "SCALABILITY" in result
    assert "AVAILABILITY" in result