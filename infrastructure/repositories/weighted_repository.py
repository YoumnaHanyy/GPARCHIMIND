from datetime import datetime

from infrastructure.database import db

collection = db["methods_results"]

def save_weighted_result(project_id, result):
    doc = {
        "project_id": project_id,
        "method": "weighted",
        "top_architectures": result["top_architectures"],
        "created_at": datetime.utcnow()
    }
    doc["method"] = "weighted"
    collection.insert_one(doc)
