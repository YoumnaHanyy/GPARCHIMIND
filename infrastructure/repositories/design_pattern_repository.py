from infrastructure.database import db
from datetime import datetime

collection = db["design_patterns"]

def save_design_patterns(project_id, patterns):
    collection.update_one(
        {"project_id": project_id},
        {
            "$set": {
                "project_id": project_id,
                "patterns": patterns,
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )