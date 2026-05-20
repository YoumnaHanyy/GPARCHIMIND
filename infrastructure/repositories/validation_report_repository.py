from infrastructure.database import db
from datetime import datetime
from bson import Binary


def save_validation_report_pdf(
    project_id: str,
    pdf_bytes: bytes
):

    db.validation_reports.delete_many({
        "project_id": project_id
    })

    db.validation_reports.insert_one({
        "project_id": project_id,
        "report_pdf": Binary(pdf_bytes),
        "created_at": datetime.utcnow()
    })