from infrastructure.database import db
from datetime import datetime
from bson import Binary

def save_architecture_report_pdf(
    project_id: str,
    pdf_bytes: bytes
):
    # Only delete the main architecture report (no report_type field).
    # Documents with report_type (e.g. "verification") are kept untouched.
    db.architecture_reports.delete_many({
        "project_id": project_id,
        "report_type": {"$exists": False}
    })

    result = db.architecture_reports.insert_one({
        "project_id": project_id,
        "report_pdf": Binary(pdf_bytes),
        "created_at": datetime.utcnow()
    })
    print(f"[ADL_repo] architecture report saved — db={db.name} "
          f"collection=architecture_reports inserted_id={result.inserted_id}", flush=True)
