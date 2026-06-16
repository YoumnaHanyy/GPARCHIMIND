from flask import Blueprint, render_template, jsonify, request
from infrastructure.database import db
from datetime import datetime

project_bp = Blueprint("project", __name__)


# ── Existing: open project dashboard page ───────────────────────────────────
@project_bp.route("/project/<project_id>")
def open_project(project_id):

    project = db.projects.find_one(
        {"project_id": project_id},
        {"_id": 0}
    )

    frs = list(db.fr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))

    nfrs = list(db.nfr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))

    hybrid_results        = project.get("hybrid_method", [])          if project else []
    selected_architecture = project.get("selectedArchitecture", "Unknown") if project else "Unknown"
    architecture_history  = project.get("architecture_history", [])   if project else []

    return render_template(
        "project_dashboard.html",
        project=project,
        frs=frs,
        nfrs=nfrs,
        hybrid_results=hybrid_results,
        selected_architecture=selected_architecture,
        architecture_history=architecture_history,
    )


# ── NEW: Load full project data as JSON for the Continue Project resume flow ─
@project_bp.route("/projects/load/<project_id>")
def load_project_data(project_id):
    """
    Returns full saved project state as JSON so Dashboard.js can hydrate
    extractedData and resume from the saved phase without creating a new project.
    """
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    project = db.projects.find_one(
        {"project_id": project_id},
        {"_id": 0}
    )

    if not project:
        return jsonify({"error": "Project not found"}), 404

    frs = list(db.fr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))

    nfrs = list(db.nfr_extracted.find(
        {"project_id": project_id},
        {"_id": 0}
    ))

    payload = {
        "project_id":           project.get("project_id"),
        "project_name":         project.get("project_name"),
        "current_phase":        project.get("current_phase", 1),
        "progress":             project.get("progress", 0),
        "status":               project.get("status"),

        # Requirements
        "functional":           frs,
        "nfr_predictions":      nfrs,

        # Architecture results (None if not reached yet)
        "functional_method":    project.get("functional_method"),
        "ordinal_method":       project.get("ordinal_method"),
        "binary_method":        project.get("binary_method"),
        "weighted_method":      project.get("weighted_method"),
        "hybrid_method":        project.get("hybrid_method"),

        "selectedArchitecture": project.get("selectedArchitecture"),
    }

    return jsonify(payload), 200


# ── Existing: save updated requirements ─────────────────────────────────────
@project_bp.route("/project/<project_id>/save-updates", methods=["POST"])
def save_updates(project_id):
    body = request.get_json()

    db.fr_extracted.delete_many({"project_id": project_id})
    db.nfr_extracted.delete_many({"project_id": project_id})

    for fr in body.get("functional", []):
        db.fr_extracted.insert_one({**fr, "project_id": project_id})

    for nfr in body.get("nfr_predictions", []):
        db.nfr_extracted.insert_one({**nfr, "project_id": project_id})

    return jsonify({"status": "saved"}), 200


# ── Existing: update project progress ───────────────────────────────────────
@project_bp.route("/projects/update-progress", methods=["POST"])
def update_progress():
    body = request.get_json()

    db.projects.update_one(
        {"project_id": body["project_id"]},
        {
            "$set": {
                "progress":     body["progress"],
                "current_phase": body["phase"],
                "updated_at":   datetime.utcnow()
            }
        }
    )

    return jsonify({"status": "updated"}), 200


# ── Existing: save updated architectures after re-evaluation ────────────────
@project_bp.route("/project/<project_id>/save-architectures", methods=["POST"])
def save_architectures(project_id):
    body = request.get_json()

    db.projects.update_one(
        {"project_id": project_id},
        {
            "$set": {
                "hybrid_method":        body.get("hybrid_method"),
                "selectedArchitecture": body.get("selected_architecture"),
                "updated_at":           datetime.utcnow()
            }
        }
    )

    return jsonify({"status": "saved"}), 200


# ── Existing: re-evaluate architecture ──────────────────────────────────────
@project_bp.route("/project/<project_id>/reevaluate", methods=["POST"])
def reevaluate(project_id):
    from ai.facades.architecture_facade import ArchitectureFacade

    body = request.get_json()
    frs  = body.get("frs",  [])
    nfrs = body.get("nfrs", [])

    facade = ArchitectureFacade()
    result = facade.evaluate(
        project_id=project_id,
        frs=frs,
        nfrs=nfrs
    )

    return jsonify(result), 200