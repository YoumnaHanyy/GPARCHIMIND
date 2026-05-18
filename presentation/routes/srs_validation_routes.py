"""
ArchiMind — SRS Validation Routes
=====================================
Endpoints:
    GET    /srs-validations                        → dashboard page
    POST   /validate-srs                           → run full validation pipeline
    GET    /validation-project/{id}               → load existing validation
    DELETE /validation-project/{id}               → delete a validation
    GET    /download-enhanced-srs/{id}            → download Enhanced SRS as PDF

All API endpoints return:
    { "success": bool, ...payload }
"""

import os
import uuid
import logging
import traceback

from fastapi import APIRouter, UploadFile, Request, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import io

from service.srs_validation_service import (
    validate_srs_pipeline,
    ValidationError,
    EmptyDocumentError,
    InsufficientContentError,
)
from infrastructure.repositories.validation_repository import (
    create_validation_project,
    save_validation_results,
    mark_validation_failed,
    get_user_validation_projects,
    get_validation_project,
    delete_validation_project,
)
from ai.validations.enhanced_srs_pdf import generate_enhanced_srs_pdf


logger = logging.getLogger(__name__)

router    = APIRouter()
templates = Jinja2Templates(directory="presentation/templates")

UPLOAD_DIR   = "uploads"
MAX_FILE_MB  = 20
ALLOWED_TYPE = "application/pdf"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _get_current_user(request: Request) -> dict | None:
    return request.session.get("user")


def _require_auth(request: Request) -> dict:
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _error_response(message: str, status_code: int = 400, **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message, **extra}
    )


# ──────────────────────────────────────────────
# DASHBOARD PAGE
# ──────────────────────────────────────────────

@router.get("/srs-validations", response_class=HTMLResponse)
async def srs_validations_page(request: Request):

    user = _get_current_user(request)

    if not user:
        return templates.TemplateResponse(
            "Login.html",
            {"request": request}
        )

    user_id  = user.get("id")
    projects = get_user_validation_projects(user_id)

    return templates.TemplateResponse(
        "srs_validation.html",
        {
            "request":  request,
            "projects": projects,
            "user":     user,
        }
    )


# ──────────────────────────────────────────────
# VALIDATE SRS  (main pipeline)
# ──────────────────────────────────────────────

@router.post("/validate-srs")
async def validate_srs(
    request: Request,
    file: UploadFile = File(...)
):
    # ── Auth ───────────────────────────────────
    user    = _get_current_user(request)
    user_id = user.get("id", "guest") if user else "guest"

    # ── Input validation ───────────────────────
    if not file or not file.filename:
        return _error_response("No file uploaded.")

    if not file.filename.lower().endswith(".pdf"):
        return _error_response(
            "Only PDF files are accepted.",
            code="INVALID_FORMAT"
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        return _error_response("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return _error_response(
            f"File exceeds the {MAX_FILE_MB} MB limit.",
            code="FILE_TOO_LARGE"
        )

    # ── Persist temp file ─────────────────────
    validation_id = f"val_{uuid.uuid4().hex[:10]}"
    pdf_path      = os.path.join(UPLOAD_DIR, f"{validation_id}.pdf")

    with open(pdf_path, "wb") as fh:
        fh.write(file_bytes)

    # ── Create DB record (status = processing) ─
    create_validation_project(
        validation_id=validation_id,
        user_id=user_id,
        project_name="Processing...",
        file_name=file.filename,
    )

    # ── Run pipeline ───────────────────────────
    try:
        result = validate_srs_pipeline(
            pdf_path=pdf_path,
            validation_id=validation_id,
            original_filename=file.filename,
        )
    except EmptyDocumentError as exc:
        mark_validation_failed(validation_id, str(exc))
        return _error_response(
            "Could not extract text from the PDF. "
            "Ensure the document is not scanned/image-only.",
            status_code=422
        )
    except InsufficientContentError as exc:
        mark_validation_failed(validation_id, str(exc))
        return _error_response(
            "The document contains too little content to validate meaningfully.",
            status_code=422
        )
    except ValidationError as exc:
        mark_validation_failed(validation_id, str(exc))
        logger.error("Validation pipeline error | id=%s: %s", validation_id, exc)
        return _error_response(str(exc), status_code=500)
    except Exception as exc:
        mark_validation_failed(validation_id, str(exc))
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1] if tb else None
        logger.exception("Unexpected error | id=%s", validation_id)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected error occurred during validation.",
                "detail":  str(exc),
                "location": {
                    "file": os.path.relpath(last.filename) if last else "unknown",
                    "line": last.lineno if last else 0,
                    "code": last.line   if last else "",
                },
            }
        )

    # ── Persist results ────────────────────────
    save_validation_results(validation_id, result)

    return {
        "success":       True,
        "validation_id": validation_id,
        **result,
    }


# ──────────────────────────────────────────────
# LOAD EXISTING VALIDATION
# ──────────────────────────────────────────────

@router.get("/validation-project/{validation_id}")
async def open_validation_project(
    validation_id: str,
    request: Request
):
    user = _get_current_user(request)

    project = get_validation_project(validation_id)

    if not project:
        return _error_response(
            "Validation project not found.",
            status_code=404
        )

    # Optional: verify ownership (if not guest)
    if user and project.get("user_id") not in (user.get("id"), "guest"):
        return _error_response(
            "You do not have access to this validation project.",
            status_code=403
        )

    results = project.get("results") or {}

    return {
        "success":       True,
        "validation_id": validation_id,
        "project_name":  project.get("project_name"),
        "file_name":     project.get("file_name"),
        "status":        project.get("status"),
        "created_at":    str(project.get("created_at", "")),
        **results,
    }


# ──────────────────────────────────────────────
# DELETE VALIDATION
# ──────────────────────────────────────────────

@router.delete("/validation-project/{validation_id}")
async def delete_validation(
    validation_id: str,
    request: Request
):
    user = _require_auth(request)
    user_id = user.get("id")

    deleted = delete_validation_project(validation_id, user_id)

    if not deleted:
        return _error_response(
            "Validation project not found or access denied.",
            status_code=404
        )

    return {"success": True, "message": "Validation project deleted."}


# ──────────────────────────────────────────────
# DOWNLOAD ENHANCED SRS AS PDF
# ──────────────────────────────────────────────

@router.get("/download-enhanced-srs/{validation_id}")
async def download_enhanced_srs(
    validation_id: str,
    request: Request
):
    """
    Generate and stream the Enhanced SRS as a downloadable PDF.
    """
    project = get_validation_project(validation_id)

    if not project:
        return _error_response(
            "Validation project not found.",
            status_code=404
        )

    results = project.get("results")
    if not results:
        return _error_response(
            "Validation results not available yet.",
            status_code=404
        )

    try:
        pdf_bytes = generate_enhanced_srs_pdf(results)

        project_name = results.get("project_name", "Enhanced_SRS")
        # Sanitize filename
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in project_name
        )
        filename = f"Enhanced_SRS_{safe_name}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        logger.exception("PDF generation failed | id=%s", validation_id)
        return _error_response(
            f"Failed to generate PDF: {str(e)}",
            status_code=500
        )