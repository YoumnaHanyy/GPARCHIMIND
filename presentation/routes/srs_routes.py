from fastapi import APIRouter, UploadFile,Request, File
from fastapi.responses import JSONResponse
import os
import uuid
import traceback

from application.extraction.extraction_service import process_srs
from ai.inference.predict_type_level import predict_and_save_nfr
from service.ordinal_service import execute_ordinal_method
from service.binary_service import execute_binary_method
from service.weighted_service import execute_weighted_method
from service.nfr_stats_service import compute_nfr_statistics
from service.functional_service import execute_functional_method
from service.hybrid_service import execute_hybrid_method
from infrastructure.repositories.project_repo import update_project_progress
from infrastructure.repositories.project_repo import create_project
from infrastructure.repositories.weighted_repository import save_weighted_result
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def clean_object_id(items: list):
    cleaned = []
    for item in items:
        item = dict(item)
        item.pop("_id", None)
        item.pop("project_id", None)
        cleaned.append(item)
    return cleaned


@router.post("/extract")
async def extract_srs(
    request: Request,
    file: UploadFile = File(...)
):
    project_id = f"proj_{uuid.uuid4().hex[:8]}"

    try:
        # =========================
        # 0️⃣ validate input
        # =========================
        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"}
            )

        # =========================
        # 1️⃣ save pdf (unique name)
        # =========================
        pdf_path = os.path.join(UPLOAD_DIR, f"{project_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # =========================
        # 2️⃣ extract FR + NFR
        # =========================
        extraction_result = process_srs(
            pdf_path=pdf_path,
            project_id=project_id,
            hf_key=os.getenv("HF_API_KEY")
        )

        if not extraction_result:
            raise ValueError("process_srs returned empty result")

        # =========================
        # 3️⃣ functional architecture
        # =========================
        functional_result = execute_functional_method(project_id)

        # =========================
        # 4️⃣ predict NFR
        # =========================
        predictions = predict_and_save_nfr()

        if not predictions:
            raise ValueError("No NFR predictions generated")

        # =========================
        # 5️⃣ statistics for weighted
        # =========================
        freq_norm, must_norm, importance = compute_nfr_statistics(predictions)

        # =========================
        # 6️⃣ methods
        # =========================
        ordinal_result = execute_ordinal_method()
        binary_result = execute_binary_method()

        weighted_result = execute_weighted_method(
            freq_norm=freq_norm,
            must_norm=must_norm,
            importance=importance
        )

        hybrid_result = execute_hybrid_method(
            project_id,
            functional_result,
            ordinal_result,
            binary_result,
            weighted_result
        )

        # =========================
        # 7️⃣ persistence
        # =========================
        save_weighted_result(project_id, weighted_result)

        user_id = request.session.get("user", {}).get("id", "guest")
        create_project(project_id, user_id)

        # =========================
        # 8️⃣ response
        # =========================
        return {
            "project_id": project_id,
            "functional": clean_object_id(extraction_result.get("functional", [])),
            "nfr_predictions": clean_object_id(predictions),
            "functional_method": functional_result,
            "ordinal_method": ordinal_result.get("result"),
            "binary_method": binary_result,
            "weighted_method": weighted_result,
            "hybrid_method": hybrid_result
        }

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1]

        print("❌ EXTRACT FAILED:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "message": "Extraction pipeline failed",
                "error": str(e),
                "location": {
                    "file": os.path.relpath(last.filename),
                    "line": last.lineno,
                    "code": last.line
                }
            }
        )


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse({"message": "Logged out"})

@router.post("/projects/update-progress")
async def update_progress(request: Request):
    body = await request.json()

    project_id = body.get("project_id")
    phase = body.get("phase")
    progress = body.get("progress")

    if not project_id:
        return {"error": "project_id missing"}

    update_project_progress(
        project_id=project_id,
        progress=progress,
        phase=phase
    )

    return {"status": "ok"}
