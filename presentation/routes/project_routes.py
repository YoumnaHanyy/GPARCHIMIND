# presentation/routes/project_routes.py

from flask import Blueprint, render_template
from infrastructure.database import db
from infrastructure.repositories.code_skeleton_repository import (get_code_skeleton)

project_bp = Blueprint("project", __name__)

@project_bp.route("/project/<project_id>")
def open_project(project_id):

    project = db.projects.find_one(
        {"project_id": project_id}
    )

    frs = list(db.fr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))

    nfrs = list(db.nfr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))
    patterns_doc = db.design_patterns.find_one(
    {"project_id": project_id})

    patterns = []

    if patterns_doc:
       patterns = patterns_doc.get("patterns", [])


    skeleton = get_code_skeleton(project_id)


    adl_report_exists = db.architecture_reports.find_one({
       "project_id": project_id})

    validation_report_exists = db.validation_reports.find_one({
        "project_id": project_id})

    verification_report_exists = db.ADLVerificationReports.find_one({
        "project_id": project_id})

    return render_template(
        "project_dashboard.html",
        project=project,
        frs=frs,
        nfrs=nfrs,
        patterns=patterns,
        skeleton=skeleton,

        adl_report_exists=bool(adl_report_exists),
        validation_report_exists=bool(validation_report_exists),
        verification_report_exists=bool(verification_report_exists)
    )