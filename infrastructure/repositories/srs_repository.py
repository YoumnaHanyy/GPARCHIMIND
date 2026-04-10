from datetime import datetime
from infrastructure.database import db, fs


class SRSRepository:

    @staticmethod
    def save_srs(file_bytes, filename, project_id, user_id, num_pages, status):
        # 1️⃣ save file in GridFS
        file_id = fs.put(file_bytes, filename=filename)

        # 2️⃣ save metadata
        doc = {
            "project_id": project_id,
            "user_id": user_id,
            "filename": filename,
            "file_id": file_id,
            "num_pages": num_pages,
            "status": status,
            "uploaded_at": datetime.utcnow()
        }

        db.srs_documents.insert_one(doc)

        return file_id