from ai.inference.predict_type_level import predict_and_save_nfr


def test_predict_and_save_nfr_integration(mocker):
    # ============================
    # 1️⃣ Fake dataset (for encoder)
    # ============================
    import pandas as pd

    fake_df = pd.DataFrame({
        "Type": ["SE", "SC"],
        "Level": ["High", "Medium"]
    })

    mocker.patch(
        "ai.inference.predict_type_level.NFRDatasetRepository.load_nfr_dataset_from_mongo",
        return_value=fake_df
    )

    # ============================
    # 2️⃣ Fake JSON input file
    # ============================
    fake_nfrs = [
        {"title": "Security", "description": "The system must be secure"},
        {"title": "Scalability", "description": "The system must scale"}
    ]

    mock_file = mocker.mock_open(read_data="dummy")
    mocker.patch("builtins.open", mock_file)

    mocker.patch(
        "ai.inference.predict_type_level.json.load",
        return_value=fake_nfrs
    )

    # ============================
    # 3️⃣ Mock tokenizer + model
    # ============================
    class FakeModel:
        def eval(self): pass

        def __call__(self, **kwargs):
            class Output:
                logits = __import__("torch").tensor([[0.9, 0.1], [0.2, 0.8]])
            return Output()

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": None, "attention_mask": None}

    mocker.patch(
        "ai.inference.predict_type_level.BertTokenizer.from_pretrained",
        return_value=FakeTokenizer()
    )

    mocker.patch(
        "ai.inference.predict_type_level.BertForSequenceClassification.from_pretrained",
        return_value=FakeModel()
    )

    # ============================
    # 4️⃣ Mock saving to Mongo
    # ============================
    mock_save = mocker.patch(
        "ai.inference.predict_type_level.NFRPredictionRepository.save_batch"
    )

    # ============================
    # 5️⃣ Run function
    # ============================
    result = predict_and_save_nfr("project1")

    # ============================
    # 6️⃣ Assertions
    # ============================
    assert isinstance(result, list)
    assert len(result) == 2

    for item in result:
        assert "predicted_type" in item
        assert "predicted_level" in item
        assert "confidence" in item

    mock_save.assert_called_once()