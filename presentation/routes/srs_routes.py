from fastapi import APIRouter, UploadFile, Request, File
from fastapi.responses import JSONResponse

import os
import uuid
import traceback
import fitz
import pdfplumber
import traceback

from ai.inference.feature_extractor import generate_phase4
from application.extraction.extraction_service import process_srs
from ai.inference.predict_type_level import predict_and_save_nfr, predict_level_for_text
from service.ordinal_service import execute_ordinal_method
from service.binary_service import execute_binary_method
from service.weighted_service import execute_weighted_method
from service.nfr_stats_service import compute_nfr_statistics
from service.functional_service import execute_functional_method
from service.hybrid_service import execute_hybrid_method
from infrastructure.repositories.project_repo import update_project_progress, create_project
from infrastructure.repositories.weighted_repository import save_weighted_result
from infrastructure.repositories.nfr_dataset_repository import NFRPredictionRepository
from infrastructure.repositories.srs_repository import SRSRepository
from infrastructure.repositories.human_feedback_repository import save_new_confirmed_nfr
from service.retrain_service import merge_and_retrain
from infrastructure.database import db
from service.retrain_service import run_retrain_async
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_pdf_file(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        return "Invalid file format. Please upload a valid PDF document."
    return None

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def check_srs_sections(text: str):
    text = text.lower()

    has_functional = (
        "functional requirements" in text or
        "functional requirement" in text
    )

    has_non_functional = (
        "non-functional requirements" in text or
        "non functional requirements" in text or
        "non-functional requirement" in text
    )

    return has_functional, has_non_functional


def clean_object_id(items: list):
    """Remove MongoDB-specific fields from response"""
    cleaned = []
    for item in items:
        item = dict(item)
        item.pop("_id", None)
        cleaned.append(item)
    return cleaned


@router.post("/extract")
async def extract_srs(request: Request, file: UploadFile = File(...)):
    project_id = f"proj_{uuid.uuid4().hex[:8]}"

    try:
        if not file:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        # ✅ USE VALIDATION FUNCTION
        validation_error = validate_pdf_file(file)
        if validation_error:
            return JSONResponse(
                status_code=400,
                content={"error": validation_error}
            )

        # 1️⃣ Save PDF
        pdf_path = os.path.join(UPLOAD_DIR, f"{project_id}.pdf")
        file_bytes = await file.read()

        with open(pdf_path, "wb") as f:
          f.write(file_bytes)
        
        # ✅ STRUCTURE CHECK (HERE ONLY)
        pdf_text = extract_text_from_pdf(pdf_path)

        pdf_doc = fitz.open(pdf_path)
        num_pages = len(pdf_doc)
        pdf_doc.close()
        has_fr, has_nfr = check_srs_sections(pdf_text)
        user_id = request.session.get("user", {}).get("id", "guest")
        status = "verified" if (has_fr and has_nfr) else "error"
        SRSRepository.save_srs(
              file_bytes=file_bytes,
              filename=file.filename,
              project_id=project_id,
              user_id=user_id,
              num_pages=num_pages,
              status=status
        )

        if not has_fr or not has_nfr:
         return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid SRS format",
            "message": (
                "The uploaded SRS must contain clearly defined "
                "'Functional Requirements' and 'Non-Functional Requirements' sections."
            )
        }
    )

        # 2️⃣ Extract FR + NFR
        extraction_result = process_srs(
            pdf_path=pdf_path,
            project_id=project_id,
            hf_key=None
        )
        
        if not extraction_result:
            raise ValueError("process_srs returned empty result")
        project_name = extraction_result.get("project_name", "Unknown Project")
        user_id = request.session.get("user", {}).get("id", "guest")

        create_project(project_id, user_id, project_name)
        # 3️⃣ Predict NFR Type + Level → Saves to BOTH MongoDB AND JSON
        all_predictions =predict_and_save_nfr(project_id)

        if not all_predictions:
            raise ValueError("No NFR predictions generated")

        # 4️⃣ Get high and low confidence from MongoDB
        high_confidence = NFRPredictionRepository.get_high_confidence(project_id)
        low_confidence = NFRPredictionRepository.get_low_confidence(project_id)
        print(f"📊 High confidence: {len(high_confidence)}, Low confidence: {len(low_confidence)}")
        if len(low_confidence) == 0:
            print("⚡ No low confidence → auto running architecture")

            all_nfrs = high_confidence

            functional_result = execute_functional_method(project_id)
    
            freq_norm, must_norm, importance = compute_nfr_statistics(all_nfrs)
    
            ordinal_result = execute_ordinal_method(project_id)
            binary_result = execute_binary_method(project_id)
    
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

            save_weighted_result(project_id, weighted_result)

            return {
                "project_id": project_id,
                "srs_verified": True,
                "functional": clean_object_id(extraction_result.get("functional", [])),
                "nfr_predictions": clean_object_id(high_confidence),
                "low_confidence_nfrs": [],
                "needs_confirmation": False,
                "functional_method": functional_result,
                "ordinal_method": ordinal_result.get("result"),
                "binary_method": binary_result,
                "weighted_method": weighted_result,
                "hybrid_method": hybrid_result
            }
        

        # 5️⃣ Return data to frontend
        return {
            "project_id": project_id,
            "srs_verified":True,
            "functional": clean_object_id(extraction_result.get("functional", [])),
            "nfr_predictions": clean_object_id(high_confidence),
            "low_confidence_nfrs": clean_object_id(low_confidence),
            "needs_confirmation": len(low_confidence) > 0
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

@router.post("/generate-skeleton")
async def generate_skeleton(request: Request):

    body = await request.json()

    project_id = body.get("project_id")
    language = body.get("language", "python")

    architecture = load_selected_architecture(project_id)

    frs, nfrs = load_requirements()

    patterns_doc = db.design_patterns.find_one({
        "project_id": project_id
    })

    patterns = []

    if patterns_doc:
        patterns = patterns_doc.get("patterns", [])

    code = generate_code_skeleton(
        architecture=architecture,
        functional=frs,
        nfrs=nfrs,
        patterns=patterns,
        language=language
    )
    save_code_skeleton(
    project_id=project_id,
    language=language,
    tree=code
)
    return {
        "code": code
    }
@router.post("/confirm_nfr")
async def confirm_nfr(request: Request):
    """
    Called after user confirms low confidence NFRs
    Updates BOTH MongoDB and JSON file, then triggers architecture analysis
    """
    body = await request.json()

    project_id = body.get("project_id")
    items = body.get("items", [])

    if not project_id:
        return JSONResponse(
            status_code=400,
            content={"error": "project_id is required"}
        )

    if not items:
        return JSONResponse(
            status_code=400,
            content={"error": "No NFR items provided"}
        )

    confirmed_count = 0

    for it in items:
        description = it.get("description")
        confirmed_type = it.get("type")

        if not description or not confirmed_type:
            continue

        # Predict level for confirmed type
        predicted_level = predict_level_for_text(description)

        # Update BOTH MongoDB and JSON file
        success = NFRPredictionRepository.confirm_nfr(
            project_id=project_id,
            description=description,
            confirmed_type=confirmed_type,
            predicted_level=predicted_level
        )

        if success:
           confirmed_count += 1
           save_new_confirmed_nfr({
                "description": description,
                "confirmed_type": confirmed_type,
                "predicted_level": predicted_level
        })
    
    count = db.new_nfr_confirmed.count_documents({})

    print(f"📊 New feedback count: {count}")

    if count >= 3:
        run_retrain_async()
    # Get ALL predictions from MongoDB (high confidence + confirmed)
    all_nfrs = NFRPredictionRepository.get_by_project(project_id)

    print(f"📊 Total NFRs after confirmation: {len(all_nfrs)}")

    # Now run architecture analysis with complete NFR set
    functional_result = execute_functional_method(project_id)
    
    freq_norm, must_norm, importance = compute_nfr_statistics(all_nfrs)
    
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

    save_weighted_result(project_id, weighted_result)

    user_id = request.session.get("user", {}).get("id", "guest")
    create_project(project_id, user_id)

    return {
        "status": "ok",
        "saved_count": confirmed_count,
        "functional_method": functional_result,
        "ordinal_method": ordinal_result.get("result"),
        "binary_method": binary_result,
        "weighted_method": weighted_result,
        "hybrid_method": hybrid_result,
        "nfr_predictions": clean_object_id(all_nfrs),
       
    }


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