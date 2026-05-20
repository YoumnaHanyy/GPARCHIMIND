"""
ArchiMind — Validation Repository
===================================
MongoDB schema for validation_projects collection.

Document Structure:
{
    "validation_id":      str,          # "val_abc12345"
    "user_id":            str,
    "project_name":       str,
    "status":             str,          # "pending" | "processing" | "completed" | "failed"
    "progress":           int,          # 0-100  (stored as quality_score after completion)
    "created_at":         datetime,
    "updated_at":         datetime,
    "file_name":          str,          # original filename
    "results": {
        "project_name":       str,
        "total_requirements": int,
        "issues":             list[dict],
        "critical_issues":    int,
        "high_issues":        int,
        "medium_issues":      int,
        "low_issues":         int,
        "quality_score":      float,
        "compliance":         dict,
        "sections_found":     list[str],
        "sections_missing":   list[str],
        "suggestions":        list[str],
        "enhanced_srs":       str,
        "ai_summary":         str,
    }
}
"""

from infrastructure.database import db
from datetime import datetime


validation_collection = db["validation_projects"]


# ──────────────────────────────────────────────
# CREATE
# ──────────────────────────────────────────────

def create_validation_project(
    validation_id: str,
    user_id: str,
    project_name: str,
    file_name: str = "unknown.pdf"
) -> None:
    """
    Insert a new validation project document with 'processing' status.
    Call this BEFORE running the validation pipeline.
    """
    validation_collection.insert_one({
        "validation_id": validation_id,
        "user_id":        user_id,
        "project_name":   project_name,
        "file_name":      file_name,
        "status":         "processing",
        "progress":       0,
        "created_at":     datetime.utcnow(),
        "updated_at":     datetime.utcnow(),
        "results":        None,
    })


# ──────────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────────

def save_validation_results(
    validation_id: str,
    results: dict
) -> None:
    """
    Persist the validation pipeline output.
    Sets status = 'completed' and progress = quality_score.
    """
    quality_score = results.get("quality_score", 0)

    validation_collection.update_one(
        {"validation_id": validation_id},
        {
            "$set": {
                "results":      results,
                "status":       "completed",
                "progress":     int(quality_score),
                "project_name": results.get("project_name", "Unknown Project"),
                "updated_at":   datetime.utcnow(),
            }
        }
    )


# ──────────────────────────────────────────────
# MARK FAILED
# ──────────────────────────────────────────────

def mark_validation_failed(
    validation_id: str,
    error_message: str
) -> None:
    """Mark a validation as failed with the error reason."""
    validation_collection.update_one(
        {"validation_id": validation_id},
        {
            "$set": {
                "status":       "failed",
                "error":        error_message,
                "updated_at":   datetime.utcnow(),
            }
        }
    )


# ──────────────────────────────────────────────
# READ — SINGLE PROJECT
# ──────────────────────────────────────────────

def get_validation_project(validation_id: str) -> dict | None:
    """
    Return a single validation project by its ID.
    Excludes MongoDB's internal _id field.
    """
    return validation_collection.find_one(
        {"validation_id": validation_id},
        {"_id": 0}
    )


# ──────────────────────────────────────────────
# READ — USER PROJECTS
# ──────────────────────────────────────────────

def get_user_validation_projects(user_id: str) -> list[dict]:
    """
    Return all validation projects for a user, newest first.
    Only returns summary fields for the dashboard (no full results payload).
    """
    return list(
        validation_collection.find(
            {"user_id": user_id},
            {
                "_id":           0,
                "validation_id": 1,
                "project_name":  1,
                "file_name":     1,
                "status":        1,
                "progress":      1,
                "created_at":    1,
                "updated_at":    1,
                # Lightweight summary fields from results
                "results.quality_score":      1,
                "results.total_requirements": 1,
                "results.critical_issues":    1,
            }
        ).sort("created_at", -1)
    )


# ──────────────────────────────────────────────
# DELETE
# ──────────────────────────────────────────────

def delete_validation_project(
    validation_id: str,
    user_id: str
) -> bool:
    """
    Delete a validation project (only if it belongs to the user).
    Returns True if deleted, False if not found.
    """
    result = validation_collection.delete_one({
        "validation_id": validation_id,
        "user_id":        user_id,
    })
    return result.deleted_count > 0


# ──────────────────────────────────────────────
# STATS (optional utility)
# ──────────────────────────────────────────────

def get_user_validation_stats(user_id: str) -> dict:
    """Return aggregate statistics for a user's validations."""
    pipeline = [
        {"$match": {"user_id": user_id, "status": "completed"}},
        {
            "$group": {
                "_id":               None,
                "total_validations": {"$sum": 1},
                "avg_quality_score": {"$avg": "$progress"},
                "max_quality_score": {"$max": "$progress"},
                "min_quality_score": {"$min": "$progress"},
            }
        }
    ]
    result = list(validation_collection.aggregate(pipeline))
    if result:
        stats = result[0]
        stats.pop("_id", None)
        stats["avg_quality_score"] = round(stats.get("avg_quality_score", 0), 1)
        return stats
    return {
        "total_validations": 0,
        "avg_quality_score": 0,
        "max_quality_score": 0,
        "min_quality_score": 0,
    }