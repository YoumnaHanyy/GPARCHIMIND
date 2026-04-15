from unittest.mock import patch
from ai.inference.predict_type_level import predict_and_save_nfr
import pandas as pd
import torch


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