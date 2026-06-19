from infrastructure.database import db
from datetime import datetime
from bson import Binary


def save_verification_report_pdf1(
    project_id: str,
    pdf_bytes: bytes
):
    db.ADLVerificationReports.delete_many({
        "project_id": project_id
    })

    db.ADLVerificationReports.insert_one({
        "project_id": project_id,
        "report_pdf": Binary(pdf_bytes),
        "created_at": datetime.utcnow()
    })