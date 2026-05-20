from infrastructure.database import db
from datetime import datetime

collection = db["methods_results"]

def save_binary_result(data: dict):
    return binary_collection.insert_one({
        "method": data["method"],
        "result": data["result"],
        "created_at": datetime.utcnow()
    }).inserted_id
