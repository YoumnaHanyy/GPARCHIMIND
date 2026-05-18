# presentation/routes/project_routes.py

from flask import Blueprint, render_template
from infrastructure.database import db

project_bp = Blueprint("project", __name__)

@project_bp.route("/project/<int:project_id>")
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

    return render_template(
        "project_dashboard.html",
        project=project,
        frs=frs,
        nfrs=nfrs
    )