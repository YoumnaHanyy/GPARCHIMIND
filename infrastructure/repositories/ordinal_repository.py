from infrastructure.database import db
from datetime import datetime

# collection name
collection = db["methods_results"]


def save_ordinal_result(project_id: str, data: dict):
    """
    Save ordinal method result to MongoDB
    """
    document = {
        "project_id": project_id,
        "method": data.get("method", "ordinal"),
        "result": data.get("result", []),
        "created_at": datetime.utcnow()
    }

    document["method"] = "ordinal"
    res = collection.insert_one(document)
    return res.inserted_id


def get_latest_ordinal_result():
    """
    Get latest ordinal method result
    """
    return collection.find_one(
        sort=[("created_at", -1)]
    )
