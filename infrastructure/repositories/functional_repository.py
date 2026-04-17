from datetime import datetime

from infrastructure.database import db

collection = db["methods_results"]

def save_functional_method(project_id, result):
    collection.delete_many({
    "project_id": project_id,
    "method": "functional"
    })

    collection.insert_one({
    "project_id": project_id,
    "method": "functional",
    "result": result,
    "created_at": datetime.utcnow()
    })
