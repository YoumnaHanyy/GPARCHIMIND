from infrastructure.database import db

weighted_collection = db["weighted_method"]

def save_weighted_result(project_id, result):
    doc = {
        "project_id": project_id,
        "method": "weighted",
        "top_architectures": result["top_architectures"]
    }
    weighted_collection.insert_one(doc)
