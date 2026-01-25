import pandas as pd
from infrastructure.database import db

class NFRDatasetRepository:

    @staticmethod
    def load_nfr_dataset_from_mongo():
        collection = db.merged_NFR_cleaned_no_dots

        docs = list(collection.find({}, {
            "_id": 0,
            "Requirement": 1,
            "Type": 1,
            "Level": 1
        }))

        return pd.DataFrame(docs)


class NFRPredictionRepository:

    @staticmethod
    def save(project_id: int, predictions: list):
        db.nfr_predictions.delete_many({"project_id": project_id})

        for p in predictions:
            p["project_id"] = project_id

        db.nfr_predictions.insert_many(predictions)
