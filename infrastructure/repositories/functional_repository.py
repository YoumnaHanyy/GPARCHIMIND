from infrastructure.database import db

collection = db.functional_method

def save_functional_method(project_id, result):
    collection.delete_many({"project_id": project_id})

    collection.insert_one({
        "project_id": project_id,
        "method": "functional",
        "result": result
    })
