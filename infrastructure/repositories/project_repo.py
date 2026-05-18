from infrastructure.database import db
from datetime import datetime

projects_collection = db["projects"]

def create_project(project_id: str, user_id, project_name=None):
    doc = {
        "project_id": project_id,
        "user_id": user_id,
        "project_name": project_name,
        "status": "in_progress",
        "progress": 0,
        "current_phase": 1,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    projects_collection.insert_one(doc)


def update_project_progress(project_id: str, progress: int, phase: int):
    projects_collection.update_one(
        {"project_id": project_id},
        {
            "$set": {
                "progress": progress,
                "current_phase": phase,
                "updated_at": datetime.utcnow()
            }
        }
    )


def get_project(project_id: str):
    return projects_collection.find_one(
        {"project_id": project_id},
        {"_id": 0}
    )

def get_user_projects(user_id):
    # Fetch all projects for this user, sorted by most recent first
    return list(projects_collection.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("created_at", -1))

def save_project_data(project_id: str, data: dict):
    projects_collection.update_one(
        {"project_id": project_id},
        {
            "$set": {
                "functional": data.get("functional"),
                "nfr_predictions": data.get("nfr_predictions"),
                "selectedArchitecture": data.get("selectedArchitecture"),
                "functional_method": data.get("functional_method"),
                "ordinal_method": data.get("ordinal_method"),
                "binary_method": data.get("binary_method"),
                "weighted_method": data.get("weighted_method"),
                "hybrid_method": data.get("hybrid_method"),
                "updated_at": datetime.utcnow()
            }
        }
    )