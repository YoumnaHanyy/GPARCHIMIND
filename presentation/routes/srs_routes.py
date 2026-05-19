from datetime import datetime

from fastapi import APIRouter, UploadFile, Request, File
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, Request, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse


import os
import uuid
import traceback
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fitz
import subprocess
from pathlib import Path


from fastapi.responses import RedirectResponse
import pdfplumber
from fastapi import APIRouter, UploadFile, Request, File, Form
from fastapi.responses import FileResponse
from application.extraction.extraction_service import process_srs
from application.extraction.adl.json_to_acme import convert_to_acme
from application.extraction.adl.json_to_acme import convert_to_acme
from ai.inference.predict_type_level import predict_and_save_nfr, predict_level_for_text
from infrastructure.repositories.code_skeleton_repository import (
    save_code_skeleton
)
from infrastructure.repositories.code_skeleton_repository import (
    get_code_skeleton
)
from ai.inference.feature_extractor import load_selected_architecture, load_requirements
from application.extraction.skeleton.generator import generate_code_skeleton
from service.ordinal_service import execute_ordinal_method
from service.binary_service import execute_binary_method
from service.weighted_service import execute_weighted_method
from service.nfr_stats_service import compute_nfr_statistics
from service.functional_service import execute_functional_method
from service.hybrid_service import execute_hybrid_method
from application.extraction.adl.validation.runner import run_validation
from application.extraction.adl.validation.validation_report_generator import generate_validation_pdf
from application.extraction.extraction_service import process_srs
from application.extraction.adl.json_to_acme import convert_to_acme
from ai.inference.predict_type_level import predict_and_save_nfr, predict_level_for_text
from application.extraction.reporting.report_generator import generate_report

from service.ordinal_service import execute_ordinal_method
from service.binary_service import execute_binary_method
from service.weighted_service import execute_weighted_method
from service.nfr_stats_service import compute_nfr_statistics
from service.functional_service import execute_functional_method
from service.hybrid_service import execute_hybrid_method

from infrastructure.repositories.project_repo import update_project_progress, create_project, save_project_data
from infrastructure.repositories.weighted_repository import save_weighted_result
from infrastructure.repositories.nfr_dataset_repository import NFRPredictionRepository
from infrastructure.repositories.srs_repository import SRSRepository
from infrastructure.repositories.human_feedback_repository import save_new_confirmed_nfr
from infrastructure.repositories.project_repo import get_user_adl_projects
from infrastructure.repositories.project_repo import get_project
from infrastructure.repositories.ADL_repository import save_architecture_report_pdf
from infrastructure.repositories.validation_report_repository import save_validation_report_pdf

from infrastructure.repositories.project_repo import get_user_adl_projects
from infrastructure.repositories.project_repo import get_project
from infrastructure.repositories.ADL_repository import save_architecture_report_pdf
from infrastructure.repositories.validation_report_repository import save_validation_report_pdf


from service.retrain_service import merge_and_retrain, run_retrain_async
from infrastructure.database import db
from ai.inference.feature_extractor import generate_phase4
from ai.ai_engine import ai_generate_architecture
from infrastructure.repositories.project_repo import create_project



from ai.json_to_c4_plantuml import convert_to_c4_plantuml
from ai.json_to_process_view import convert_to_process_view
from ai.json_to_deployment_view import convert_to_deployment_view
from ai.json_to_usecase_view import convert_to_usecase_view

from infrastructure.repositories.project_repo import get_user_adl_projects
from infrastructure.repositories.project_repo import update_project_progress, create_project, save_project_data
from infrastructure.repositories.weighted_repository import save_weighted_result
from infrastructure.repositories.nfr_dataset_repository import NFRPredictionRepository
from infrastructure.repositories.srs_repository import SRSRepository
from infrastructure.repositories.human_feedback_repository import save_new_confirmed_nfr
from application.extraction.adl.validation.runner import run_validation
from application.extraction.reporting.report_generator import generate_report
from infrastructure.repositories.project_repo import get_project
from infrastructure.repositories.ADL_repository import save_architecture_report_pdf
from infrastructure.repositories.validation_report_repository import save_validation_report_pdf
from ai.ai_engine import ai_generate_architecture
from application.extraction.adl.validation.validation_report_generator import generate_validation_pdf
from service.retrain_service import merge_and_retrain, run_retrain_async
from infrastructure.database import db
from infrastructure.repositories.project_repo import create_project
from ai.json_to_c4_plantuml import convert_to_c4_plantuml
from ai.json_to_process_view import convert_to_process_view
from ai.json_to_deployment_view import convert_to_deployment_view
from ai.json_to_usecase_view import convert_to_usecase_view
from ai.inference.feature_extractor import generate_phase4

templates = Jinja2Templates(directory="presentation/templates")

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

@router.get("/get_project/{project_id}")
def get_project_api(project_id: str):
    project = db.projects.find_one({"project_id": project_id})
    print("PROJECT FROM DB:", project)

    if not project:
        return {"error": "Project not found"}

    project.pop("_id", None)
    skeleton = get_code_skeleton(project_id)

    if skeleton:

       skeleton.pop("_id", None)

       project["code_skeleton"] = skeleton.get("tree")
       project["selected_language"] = skeleton.get("language")
    return project


@router.post("/extract")
async def extract_srs(request: Request, file: UploadFile = File(...)):

    project_id = f"proj_{uuid.uuid4().hex[:8]}"

    try:

        if not file:

            return JSONResponse(

                status_code=400,

                content={
                    "error": "No file uploaded"
                }
            )
         # ✅ USE VALIDATION FUNCTION
        validation_error = validate_pdf_file(file)
        if validation_error:
            return JSONResponse(
                status_code=400,content={"error": validation_error})

        # ==========================================
        # SAVE PDF
        # ==========================================

        pdf_path = os.path.join(
            UPLOAD_DIR,
            f"{project_id}.pdf"
        )

        file_bytes = await file.read()

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # ==========================================
        # STRUCTURE CHECK
        # ==========================================

        pdf_text = extract_text_from_pdf(pdf_path)

        pdf_doc = fitz.open(pdf_path)

        num_pages = len(pdf_doc)

        pdf_doc.close()

        has_fr, has_nfr = \
            check_srs_sections(pdf_text)

        user_id = request.session.get(
            "user",
            {}
        ).get(
            "id",
            "guest"
        )

        status = \
            "verified" \
            if (has_fr and has_nfr) \
            else "error"

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

                    "error":
                        "Invalid SRS format",

                    "message":
                        (
                            "The uploaded SRS must contain "
                            "clearly defined "
                            "'Functional Requirements' "
                            "and "
                            "'Non-Functional Requirements' "
                            "sections."
                        )
                }
            )

        # ==========================================
        # EXTRACT FR + NFR
        # ==========================================

        extraction_result = process_srs(

            pdf_path=pdf_path,

            project_id=project_id,

            hf_key=None
        )

        if not extraction_result:

            raise ValueError(
                "process_srs returned empty result"
            )

        # ==========================================
        # AI REQUIREMENT DRIFT DETECTION
        # ==========================================


        # ==========================================
        # PROJECT INFO
        # ==========================================

        project_name = extraction_result.get(
            "project_name",
            "Unknown Project"
        )

        user_id = request.session.get(
            "user",
            {}
        ).get(
            "id",
            "guest"
        )

        create_project(
            project_id,
            user_id,
            project_name
        )

        # ==========================================
        # PREDICT NFR TYPES + LEVELS
        # ==========================================

        all_predictions = \
            predict_and_save_nfr(project_id)

        if not all_predictions:

            raise ValueError(
                "No NFR predictions generated"
            )

        # ==========================================
        # GET HIGH + LOW CONFIDENCE
        # ==========================================

        high_confidence = \
            NFRPredictionRepository.get_high_confidence(
                project_id
            )

        low_confidence = \
            NFRPredictionRepository.get_low_confidence(
                project_id
            )

        save_project_data(project_id, {
            "functional":
                extraction_result.get(
                    "functional",
                    []
                ),

            "nfr_predictions":
                high_confidence

                
        })

        print(
            f"📊 High confidence: "
            f"{len(high_confidence)}, "
            f"Low confidence: "
            f"{len(low_confidence)}"
        )

        # ==========================================
        # AUTO RUN ARCHITECTURE
        # ==========================================

        if len(low_confidence) == 0:

            print(
                "⚡ No low confidence "
                "→ auto running architecture"
            )

            all_nfrs = high_confidence

            functional_result = \
                execute_functional_method(
                    project_id
                )

            freq_norm, must_norm, importance = \
                compute_nfr_statistics(
                    all_nfrs
                )

            ordinal_result = \
                execute_ordinal_method(
                    project_id
                )

            # ==========================================
            # BUILD BINARY VECTOR
            # ==========================================

            binary_vector = {

                "PE": 0,
                "SC": 0,
                "MN": 0,
                "A": 0,
                "SE": 0,
                "US": 0,
                "PO": 0,
                "O": 0
            }

            for pred in all_nfrs:

                nfr_type = pred.get(
                    "predicted_type",
                    ""
                )

                if nfr_type in binary_vector:

                    binary_vector[nfr_type] = 1

            print(
                "BINARY VECTOR:",
                binary_vector
            )

            # ==========================================
            # EXECUTE METHODS
            # ==========================================

            binary_result = \
                execute_binary_method(

                    project_id,

                    binary_vector
                )

            weighted_result = \
                execute_weighted_method(

                    freq_norm=freq_norm,

                    must_norm=must_norm,

                    importance=importance
                )

            hybrid_result = \
                execute_hybrid_method(

                    project_id,

                    functional_result,

                    ordinal_result,

                    binary_result,

                    weighted_result
                )

            save_weighted_result(
                project_id,
                weighted_result
            )

            save_project_data(project_id, {

                "functional":
                    extraction_result.get(
                        "functional",
                        []
                    ),

                "nfr_predictions":
                    high_confidence,

                "functional_method":
                    functional_result,

                "ordinal_method":
                    ordinal_result.get(
                        "result"
                    ),

                "binary_method":
                    binary_result,

                "weighted_method":
                    weighted_result,

                "hybrid_method":
                    hybrid_result,

                "selectedArchitecture":
                    hybrid_result
            })

            return {

                "project_id":
                    project_id,

                "srs_verified":
                    True,

                "functional":
                    clean_object_id(
                        extraction_result.get(
                            "functional",
                            []
                        )
                    ),

                "nfr_predictions":
                    clean_object_id(
                        high_confidence
                    ),

                "low_confidence_nfrs":
                    [],

                "needs_confirmation":
                    False,

                "functional_method":
                    functional_result,

                "ordinal_method":
                    ordinal_result.get(
                        "result"
                    ),

                "binary_method":
                    binary_result,

                "weighted_method":
                    weighted_result,

                "hybrid_method":
                    hybrid_result
            }

        # ==========================================
        # RETURN LOW CONFIDENCE
        # ==========================================

        return {

            "project_id":
                project_id,

       
            "srs_verified":
                True,

            "functional":
                clean_object_id(
                    extraction_result.get(
                        "functional",
                        []
                    )
                ),

            "nfr_predictions":
                clean_object_id(
                    high_confidence
                ),

            "low_confidence_nfrs":
                clean_object_id(
                    low_confidence
                ),

            "needs_confirmation":
                len(low_confidence) > 0
        }

    except Exception as e:

        tb = traceback.extract_tb(
            e.__traceback__
        )

        last = tb[-1]

        print("❌ EXTRACT FAILED:", e)

        traceback.print_exc()

        return JSONResponse(

            status_code=500,

            content={

                "message":
                    "Extraction pipeline failed",

                "error":
                    str(e),

                "location": {

                    "file":
                        os.path.relpath(
                            last.filename
                        ),

                    "line":
                        last.lineno,

                    "code":
                        last.line
                }
            }
        )


# ==========================================
# UPLOAD UPDATED SRS
# ==========================================

@router.post("/adl/generate")
async def adl_generate(file: UploadFile = File(...), architecture: str = Form(...)):
    try:
        # 1) احفظي الملف مؤقتًا
        file_bytes = await file.read()
        temp_path = os.path.join(UPLOAD_DIR, f"adl_{uuid.uuid4().hex}.pdf")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        # 2) استخدمي نفس extraction الحقيقي
        extraction_result = process_srs(
            pdf_path=temp_path,
            project_id="temp_proj",
            hf_key=None
        )
        frs = extraction_result.get("functional", [])
        nfrs = predict_and_save_nfr("temp_proj")
        # 3) generate ADL
        adl = ai_generate_architecture(
            "UserSystem",
            frs,
            nfrs,
            architecture
        )
        print("🔥 FRS:", frs)
        print("🔥 NFRS:", nfrs)
        print("🔥 ADL:", adl)
        # (اختياري) تحويل لـ ACME
        adl_acme = convert_to_acme(adl)
        return {"adl": adl_acme}
    except Exception as e:
        return {"error": str(e)}
    


@router.post("/adl/generate-pdf")
async def adl_generate_pdf(
    request: Request,
    file: UploadFile = File(...),
    architecture: str = Form(...)
): 
    try:   

        user = request.session.get("user")
        if not user:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "User not authenticated"
                }
            )
        user_id = user["id"]
        # =========================
        # Create unique project
        # =========================
        project_id = f"adl_{uuid.uuid4().hex[:8]}"
        # =========================
        # Save uploaded file
        # =========================
        file_bytes = await file.read()
        temp_path = os.path.join(
            UPLOAD_DIR,
            f"{project_id}.pdf"
        )
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            extraction_result = process_srs(
            pdf_path=temp_path,
            project_id=project_id,
                   hf_key=None
        )
              # =========================
        # Get project name
        # =========================
        project_name = extraction_result.get(
            "project_name",
            "Architecture Project"
        )
        # =========================
        # Create project in DB
        # =========================
        create_project(
            project_id,
            user_id,
            project_name,
            "adl_reusable"
        )
        # =========================
        # Predict NFRs
        # =========================
        nfrs = predict_and_save_nfr(project_id)
        frs = extraction_result.get("functional", [])
        adl_result = ai_generate_architecture(
            project_name,
                 frs,
            nfrs,
            architecture
        )
        adl_acme = convert_to_acme(adl_result)
        # =========================
        # Save ACME file
        # =========================
        acme_path = os.path.join(
            "data",
            "outputs",
            "architecture.acme"
        )
        with open(acme_path, "w", encoding="utf-8") as f:
            f.write(adl_acme)
        # =========================
        # Save hybrid architecture result
        # =========================
        db.hybrid_method.update_one(
            {"project_id": project_id},
            {
                "$set": {
                    "project_id": project_id,
                    "selected_architecture": architecture
                }
            },
            upsert=True
        )
        # =========================
        # Save project data
        # =========================
        save_project_data(project_id, {
            "functional": frs,
            "nfr_predictions": nfrs,
            "selectedArchitecture": architecture
        })
        # =========================
        # Generate PlantUML Views
        # =========================
        c4_puml = convert_to_c4_plantuml(adl_result)
        process_puml = convert_to_process_view(adl_result)
        deployment_puml = convert_to_deployment_view(adl_result)
        usecase_puml = convert_to_usecase_view(adl_result)
        outputs_dir = Path("data/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        c4_file = outputs_dir / "architecture_c4.puml"
        process_file = outputs_dir / "process_view.puml"
        deployment_file = outputs_dir / "deployment_view.puml"
        usecase_file = outputs_dir / "usecase_view.puml"
        c4_file.write_text(c4_puml, encoding="utf-8")
        process_file.write_text(process_puml, encoding="utf-8")
        deployment_file.write_text(deployment_puml, encoding="utf-8")
        usecase_file.write_text(usecase_puml, encoding="utf-8")
        # =========================
        # Render PNGs using PlantUML
        # =========================
        PLANTUML_JAR = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "infrastructure",
            "tools",
            "plantuml.jar"
        )

        PLANTUML_JAR = os.path.abspath(PLANTUML_JAR)
        subprocess.run([
            r"C:\Program Files\Java\jdk-24\bin\java.exe",
            "-jar",
            PLANTUML_JAR,
            "-tpng",
            "data/outputs/dfd_context.puml",
            "data/outputs/process_view.puml",
            "data/outputs/deployment_view.puml",
            "data/outputs/architecture_c4.puml",
            "data/outputs/usecase_view.puml"
        ])
        # =========================
        # Generate FINAL REPORT
        # =========================
        pdf_path = generate_report(project_id)
        with open(pdf_path, "rb") as pdf_file:
              pdf_bytes = pdf_file.read()
        save_architecture_report_pdf(
            project_id,
            pdf_bytes
)
        # =========================
        # RUN VALIDATION
        # =========================
        validation_result = run_validation(
            adl_result
        )
        # =========================
        # GENERATE VALIDATION PDF
        # =========================
        validation_pdf_path = generate_validation_pdf(
            validation_result
        )
        # =========================
        # READ VALIDATION PDF
        # =========================
        with open(validation_pdf_path, "rb") as f:
            validation_pdf_bytes = f.read()
            
        # =========================
        # SAVE VALIDATION REPORT
        # =========================
        save_validation_report_pdf(
            project_id,
            validation_pdf_bytes
        )
        print("VALIDATION PDF SAVED:", project_id)
        update_project_progress(
    project_id,
    100,
    1
)
        return FileResponse(
              path=str(pdf_path),
            filename="architecture_report.pdf",
            media_type="application/pdf"  )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
    
@router.get("/adl-dashboard")
async def adl_dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/Login")
    projects = get_user_adl_projects(user["id"])
    return templates.TemplateResponse(
        "adl_dashboard.html",
        {
            "request": request,
            "projects": projects,
            "user": user
        }
    )
@router.get("/adl-generator", response_class=HTMLResponse)
async def adl_generator(request: Request):
    return templates.TemplateResponse(
        "adl_generator.html",
        {
            "request": request
        })

@router.get("/adl-project/{project_id}")
async def open_adl_project(
    request: Request,
    project_id: str
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/Login")
    project = get_project(project_id)
    if not project:
        return HTMLResponse(
            content="Project not found",
            status_code=404
        )
    return templates.TemplateResponse(
        "adl_project.html",
        {
            "request": request,
            "project": project
        }
    )
@router.get("/adl-project/{project_id}")
async def open_adl_project(
    request: Request,
    project_id: str
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/Login")
    project = get_project(project_id)
    if not project:
        return HTMLResponse(
            content="Project not found",
            status_code=404
        )
    return templates.TemplateResponse(
        "adl_project.html",
        {
            "request": request,
            "project": project
        }
    )

@router.get("/adl-project/{project_id}/download")
async def download_adl_report(project_id: str):
    report = db.architecture_reports.find_one({
        "project_id": project_id
    })
    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": "PDF not found"}
        )
    pdf_bytes = report["report_pdf"]
    temp_pdf = os.path.join(
        "data",
        "outputs",
        f"{project_id}.pdf"
    )
    with open(temp_pdf, "wb") as f:
        f.write(pdf_bytes)
    return FileResponse(
    path=temp_pdf,
    media_type="application/pdf"
    )


@router.get(
    "/adl-project/{project_id}/validation-report"
)
async def open_validation_report(project_id: str):
    report = db.validation_reports.find_one({
        "project_id": project_id
    })
    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": "Validation report not found"}
        )
    pdf_bytes = report["report_pdf"]
    temp_pdf = os.path.join(
        "data",
        "outputs",
        f"{project_id}_validation.pdf"
    )
    with open(temp_pdf, "wb") as f:
        f.write(pdf_bytes)
    return FileResponse(
        path=temp_pdf,
        media_type="application/pdf"
    )








@router.post("/upload-updated-srs/{project_id}")
async def upload_updated_srs(

    request: Request,

    project_id: str,

    file: UploadFile = File(...)
):

    try:

        # ==========================================
        # SAVE UPDATED PDF
        # ==========================================

        pdf_path = os.path.join(
            UPLOAD_DIR,
            f"{project_id}_updated.pdf"
        )

        file_bytes = await file.read()

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # ==========================================
        # EXTRACT UPDATED REQUIREMENTS
        # ==========================================

        extraction_result = process_srs(

            pdf_path=pdf_path,

            project_id=project_id,

            hf_key=None
        )

        # ==========================================
        # GET OLD PROJECT DATA
        # ==========================================

        project = db.projects.find_one({

            "project_id":
                project_id
        })

        old_frs = project.get(
            "functional",
            []
        )

        old_nfrs = project.get(
            "nfr_predictions",
            []
        )

        new_frs = extraction_result.get(
            "functional",
            []
        )

        new_nfrs = extraction_result.get(
            "non_functional",
            []
        )

        # ==========================================
        # AI DRIFT DETECTION
        # ==========================================

        from service.requirement_drift_service import (
            RequirementDriftService
        )

        drift_result = \
            RequirementDriftService.detect_changes(

                old_frs,
                old_nfrs,

                new_frs,
                new_nfrs
            )

        # ==========================================
        # SAVE DRIFT
        # ==========================================

        db.projects.update_one(

            {
                "project_id":
                    project_id
            },

            {
                "$set": {

                    "requirement_drift":
                        drift_result,

                    "updated_at":
                        datetime.utcnow()
                }
            }
        )

        # ==========================================
        # REDIRECT BACK
        # ==========================================

        return RedirectResponse(

            url=f"/project/{project_id}",

            status_code=303
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
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
    Updates BOTH MongoDB and JSON file,
    then triggers architecture analysis
    """

    body = await request.json()

    project_id = body.get("project_id")

    items = body.get("items", [])

    if not project_id:

        return JSONResponse(

            status_code=400,

            content={
                "error": "project_id is required"
            }
        )

    if not items:

        return JSONResponse(

            status_code=400,

            content={
                "error": "No NFR items provided"
            }
        )

    confirmed_count = 0

    # ======================================================
    # SAVE CONFIRMED NFRS
    # ======================================================

    for it in items:

        description = it.get("description")

        confirmed_type = it.get("type")

        if not description or not confirmed_type:
            continue

        predicted_level = predict_level_for_text(
            description
        )

        success = \
            NFRPredictionRepository.confirm_nfr(

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

    # ======================================================
    # AUTO RETRAIN
    # ======================================================

    count = db.new_nfr_confirmed.count_documents({})

    print(f"📊 New feedback count: {count}")

    if count >= 3:

        run_retrain_async()

    # ======================================================
    # GET ALL NFRS
    # ======================================================

    all_nfrs = \
        NFRPredictionRepository.get_by_project(
            project_id
        )

    print(
        f"📊 Total NFRs after confirmation: {len(all_nfrs)}"
    )

    # ======================================================
    # FUNCTIONAL METHOD
    # ======================================================

    functional_result = \
        execute_functional_method(
            project_id
        )

    # ======================================================
    # NFR STATISTICS
    # ======================================================

    freq_norm, must_norm, importance = \
        compute_nfr_statistics(
            all_nfrs
        )

    # ======================================================
    # ORDINAL METHOD
    # ======================================================

    ordinal_result = \
        execute_ordinal_method(
            project_id
        )

    # ======================================================
    # TYPE MAPPING
    # ======================================================

    TYPE_MAPPING = {

        "Performance": "PE",

        "Scalability": "SC",

        "Maintainability": "MN",

        "Availability": "A",

        "Security": "SE",

        "Usability": "US",

        "Portability": "PO",

        "Reliability": "O",

        "Modularity": "MN",

        "Interoperability": "SC"
    }

    # ======================================================
    # BUILD BINARY VECTOR
    # ======================================================

    binary_vector = {

        "PE": 0,
        "SC": 0,
        "MN": 0,
        "A": 0,
        "SE": 0,
        "US": 0,
        "PO": 0,
        "O": 0
    }

    for pred in all_nfrs:

        original_type = pred.get(
            "predicted_type",
            ""
        )

        mapped_type = TYPE_MAPPING.get(
            original_type,
            ""
        )

        if mapped_type in binary_vector:

            binary_vector[mapped_type] = 1

    print("BINARY VECTOR:", binary_vector)

    # ======================================================
    # BINARY METHOD
    # ======================================================

    binary_result = \
        execute_binary_method(

            project_id,

            binary_vector
        )

    # ======================================================
    # WEIGHTED METHOD
    # ======================================================

    weighted_result = \
        execute_weighted_method(

            freq_norm=freq_norm,

            must_norm=must_norm,

            importance=importance
        )

    save_weighted_result(

        project_id,

        weighted_result
    )

    # ======================================================
    # HYBRID METHOD
    # ======================================================

    hybrid_result = \
        execute_hybrid_method(

            project_id,

            functional_result,

            ordinal_result,

            binary_result,

            weighted_result
        )

    print("HYBRID RESULT:", hybrid_result)

    # ======================================================
    # PHASE 4
    # ======================================================

    phase4 = generate_phase4(project_id)

    # ======================================================
    # SAVE PROJECT DATA
    # ======================================================

    project_doc = \
        db.projects.find_one({

            "project_id": project_id

        }) or {}

    save_project_data(

        project_id,

        {

            "functional":
                project_doc.get(
                    "functional",
                    []
                ),

            "nfr_predictions":
                all_nfrs,

            "selectedArchitecture":
                hybrid_result,

            "functional_method":
                functional_result,

            "ordinal_method":
                ordinal_result,

            "binary_method":
                binary_result,

            "weighted_method":
                weighted_result,

            "hybrid_method":
                hybrid_result
        }
    )

    # ======================================================
    # RETURN RESPONSE
    # ======================================================

    return {

        "status": "ok",

        "saved_count":
            confirmed_count,

        "functional_method":
            functional_result,

        "ordinal_method":
            ordinal_result.get("result"),

        "binary_method":
            binary_result,

        "weighted_method":
            weighted_result,

        "hybrid_method":
            hybrid_result,

        "nfr_predictions":
            clean_object_id(all_nfrs),

        "phase4":
            phase4
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







@router.post("/adl/generate")
async def adl_generate(file: UploadFile = File(...), architecture: str = Form(...)):
    try:
        # 1) احفظي الملف مؤقتًا
        file_bytes = await file.read()
        temp_path = os.path.join(UPLOAD_DIR, f"adl_{uuid.uuid4().hex}.pdf")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # 2) استخدمي نفس extraction الحقيقي
        extraction_result = process_srs(
            pdf_path=temp_path,
            project_id="temp_proj",
            hf_key=None
        )

        frs = extraction_result.get("functional", [])
        nfrs = predict_and_save_nfr("temp_proj")
        # 3) generate ADL
        adl = ai_generate_architecture(
            "UserSystem",
            frs,
            nfrs,
            architecture
        )

        print("🔥 FRS:", frs)
        print("🔥 NFRS:", nfrs)
        print("🔥 ADL:", adl)

        # (اختياري) تحويل لـ ACME
        adl_acme = convert_to_acme(adl)

        return {"adl": adl_acme}

    except Exception as e:
        return {"error": str(e)}
    



@router.post("/adl/generate-pdf")
async def adl_generate_pdf(
    request: Request,
    file: UploadFile = File(...),
    architecture: str = Form(...)
):
    try:

        # =========================
        # Require logged-in user
        # =========================
        user = request.session.get("user")

        if not user:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "User not authenticated"
                }
            )

        user_id = user["id"]

        # =========================
        # Create unique project
        # =========================
        project_id = f"adl_{uuid.uuid4().hex[:8]}"

        # =========================
        # Save uploaded file
        # =========================
        file_bytes = await file.read()

        temp_path = os.path.join(
            UPLOAD_DIR,
            f"{project_id}.pdf"
        )

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # =========================
        # Extract SRS
        # =========================
        extraction_result = process_srs(
            pdf_path=temp_path,
            project_id=project_id,
            hf_key=None
        )

        # =========================
        # Get project name
        # =========================
        project_name = extraction_result.get(
            "project_name",
            "Architecture Project"
        )

        # =========================
        # Create project in DB
        # =========================
        create_project(
            project_id,
            user_id,
            project_name,
            "adl_reusable"

        )

        # =========================
        # Predict NFRs
        # =========================
        nfrs = predict_and_save_nfr(project_id)

        frs = extraction_result.get("functional", [])

        # =========================
        # Generate ADL
        # =========================
        adl_result = ai_generate_architecture(
            project_name,
            frs,
            nfrs,
            architecture
        )

        # =========================
        # Convert to ACME
        # =========================
        adl_acme = convert_to_acme(adl_result)

        # =========================
        # Save ACME file
        # =========================
        acme_path = os.path.join(
            "data",
            "outputs",
            "architecture.acme"
        )

        with open(acme_path, "w", encoding="utf-8") as f:
            f.write(adl_acme)

        # =========================
        # Save hybrid architecture result
        # =========================
        db.hybrid_method.update_one(
            {"project_id": project_id},
            {
                "$set": {
                    "project_id": project_id,
                    "selected_architecture": architecture
                }
            },
            upsert=True
        )

        # =========================
        # Save project data
        # =========================
        save_project_data(project_id, {
            "functional": frs,
            "nfr_predictions": nfrs,
            "selectedArchitecture": architecture
        })

        # =========================
        # Generate PlantUML Views
        # =========================
        c4_puml = convert_to_c4_plantuml(adl_result)

        process_puml = convert_to_process_view(adl_result)

        deployment_puml = convert_to_deployment_view(adl_result)

        usecase_puml = convert_to_usecase_view(adl_result)

        outputs_dir = Path("data/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        c4_file = outputs_dir / "architecture_c4.puml"
        process_file = outputs_dir / "process_view.puml"
        deployment_file = outputs_dir / "deployment_view.puml"
        usecase_file = outputs_dir / "usecase_view.puml"

        c4_file.write_text(c4_puml, encoding="utf-8")
        process_file.write_text(process_puml, encoding="utf-8")
        deployment_file.write_text(deployment_puml, encoding="utf-8")
        usecase_file.write_text(usecase_puml, encoding="utf-8")

        # =========================
        # Render PNGs using PlantUML
        # =========================
        subprocess.run([
            r"C:\Program Files\Java\jdk-24\bin\java.exe",
            "-jar",
            "C:\\plantuml\\plantuml.jar",
            "-tpng",
            "data/outputs/dfd_context.puml",
            "data/outputs/process_view.puml",
            "data/outputs/deployment_view.puml",
            "data/outputs/architecture_c4.puml",
            "data/outputs/usecase_view.puml"
        ])

        # =========================
        # Generate FINAL REPORT
        # =========================
        pdf_path = generate_report(project_id)
        with open(pdf_path, "rb") as pdf_file:
              pdf_bytes = pdf_file.read()

        save_architecture_report_pdf(
            project_id,
            pdf_bytes
)
# =========================
        # RUN VALIDATION
        # =========================
        validation_result = run_validation(
            adl_result
        )

        # =========================
        # GENERATE VALIDATION PDF
        # =========================
        validation_pdf_path = generate_validation_pdf(
            validation_result
        )

        # =========================
        # READ VALIDATION PDF
        # =========================
        with open(validation_pdf_path, "rb") as f:
            validation_pdf_bytes = f.read()
            

        # =========================
        # SAVE VALIDATION REPORT
        # =========================
        save_validation_report_pdf(
            project_id,
            validation_pdf_bytes
        )
        print("VALIDATION PDF SAVED:", project_id)
        
        update_project_progress(
    project_id,
    100,
    1
)
       

        return FileResponse(
            path=str(pdf_path),
            filename="architecture_report.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
    


@router.get("/adl-dashboard")
async def adl_dashboard(request: Request):

    user = request.session.get("user")

    if not user:
        return RedirectResponse("/Login")

    projects = get_user_adl_projects(user["id"])

    return templates.TemplateResponse(
        "adl_dashboard.html",
        {
            "request": request,
            "projects": projects,
            "user": user
        }
    )



@router.get("/adl-generator", response_class=HTMLResponse)
async def adl_generator(request: Request):

    return templates.TemplateResponse(
        "adl_generator.html",
        {
            "request": request
        }
    )



@router.get("/adl-project/{project_id}")
async def open_adl_project(
    request: Request,
    project_id: str
):

    user = request.session.get("user")

    if not user:
        return RedirectResponse("/Login")

    project = get_project(project_id)

    if not project:
        return HTMLResponse(
            content="Project not found",
            status_code=404
        )

    return templates.TemplateResponse(
        "adl_project.html",
        {
            "request": request,
            "project": project
        }
    )



@router.get("/adl-project/{project_id}")
async def open_adl_project(
    request: Request,
    project_id: str
):

    user = request.session.get("user")

    if not user:
        return RedirectResponse("/Login")

    project = get_project(project_id)

    if not project:
        return HTMLResponse(
            content="Project not found",
            status_code=404
        )

    return templates.TemplateResponse(
        "adl_project.html",
        {
            "request": request,
            "project": project
        }
    )


@router.get("/adl-project/{project_id}/download")
async def download_adl_report(project_id: str):

    report = db.architecture_reports.find_one({
        "project_id": project_id
    })

    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": "PDF not found"}
        )

    pdf_bytes = report["report_pdf"]

    temp_pdf = os.path.join(
        "data",
        "outputs",
        f"{project_id}.pdf"
    )

    with open(temp_pdf, "wb") as f:
        f.write(pdf_bytes)

    return FileResponse(
    path=temp_pdf,
    media_type="application/pdf"
)


@router.get(
    "/adl-project/{project_id}/validation-report"
)
async def open_validation_report(project_id: str):

    report = db.validation_reports.find_one({
        "project_id": project_id
    })

    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": "Validation report not found"}
        )

    pdf_bytes = report["report_pdf"]

    temp_pdf = os.path.join(
        "data",
        "outputs",
        f"{project_id}_validation.pdf"
    )

    with open(temp_pdf, "wb") as f:
        f.write(pdf_bytes)

    return FileResponse(
        path=temp_pdf,
        media_type="application/pdf"
    )