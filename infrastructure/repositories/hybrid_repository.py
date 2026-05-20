from infrastructure.database import db


def save_hybrid_result(project_id, result, selected_architecture):
    db.hybrid_method.delete_many({"project_id": project_id})

    db.hybrid_method.insert_one({
        "project_id": project_id,
        "method": "hybrid",
        "top_architectures": result,
        "selected_architecture": selected_architecture
    })