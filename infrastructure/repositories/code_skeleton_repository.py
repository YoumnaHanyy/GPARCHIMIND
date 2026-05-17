from infrastructure.database import db


def save_code_skeleton(
    project_id,
    language,
    tree
):

    db.code_skeletons.update_one(

        {
            "project_id": project_id
        },

        {
            "$set": {
                "project_id": project_id,
                "language": language,
                "tree": tree
            }
        },

        upsert=True
    )


def get_code_skeleton(project_id):

    return db.code_skeletons.find_one({
        "project_id": project_id
    })