from infrastructure.database import db
from datetime import datetime

# collection name
ordinal_collection = db["ordinal_method"]


def save_ordinal_result(data: dict):
    """
    Save ordinal method result to MongoDB
    """
    document = {
        "method": data.get("method", "ordinal"),
        "result": data.get("result", []),
        "created_at": datetime.utcnow()
    }

    res = ordinal_collection.insert_one(document)
    return res.inserted_id


def get_latest_ordinal_result():
    """
    Get latest ordinal method result
    """
    return ordinal_collection.find_one(
        sort=[("created_at", -1)]
    )
