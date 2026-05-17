from infrastructure.database import db
from datetime import datetime

collection = db["methods_results"]

def save_binary_result(data: dict):
    return collection.insert_one({
    "project_id": data.get("project_id"),
    "method": "binary",
    "result": data["result"],
    "created_at": datetime.utcnow()
}).inserted_id