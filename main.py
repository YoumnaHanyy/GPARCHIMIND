from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io
from application.extraction.adl.verification.runner import run_verification
from application.extraction.adl.verification.verification_report_generator import generate_verification_pdf
from ai.inference.feature_extractor import generate_phase4
import zipfile
from infrastructure.database import db
from infrastructure.repositories.ADL_repository import save_architecture_report_pdf, save_verification_report_pdf
from infrastructure.repositories.hybrid_repository import save_hybrid_result
from fastapi.staticfiles import StaticFiles
from ai.json_to_process_view import convert_to_process_view
from fastapi.responses import FileResponse
from ai.json_to_dfd_context import convert_to_dfd_context
from application.extraction.reporting.report_generator import generate_report
from fastapi.templating import Jinja2Templates
from ai.json_to_deployment_view import convert_to_deployment_view
from application.extraction.api.hybrid_route import router as hybrid_router
from ai.json_to_c4_plantuml import convert_to_c4_plantuml
from ai.ai_engine import ai_generate_architecture
from ai.json_to_usecase_view import convert_to_usecase_view
from application.extraction.adl.json_to_acme import convert_to_acme
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from application.extraction.adl.validation.validation_report_generator import generate_validation_pdf
from application.extraction.adl.validation.runner import run_validation
from service.srs_extractor import SRSExtractor
from presentation.routes.architecture_routes import router as architecture_router
from presentation.routes.srs_routes import router as srs_router
from presentation.routes import auth
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from infrastructure.repositories.project_repo import get_user_projects
from infrastructure.database import db
from pathlib import Path
import subprocess
from ai.ai_usecase import generate_usecase_ai
import json
# Import API routes
from fastapi.responses import HTMLResponse
from presentation.routes.architecture_routes import router as architecture_router
from presentation.routes.srs_routes import router as srs_router
from dotenv import load_dotenv
import os
import threading
import time
from service.retrain_service import merge_and_retrain
from infrastructure.database import db
from service.retrain_service import run_retrain_async

def auto_retrain_loop():
    while True:
        time.sleep(86400)

        count = db.new_nfr_confirmed.count_documents({})

        if count > 0:
            print(f"⏱️ Auto retrain → {count}")
            run_retrain_async()


threading.Thread(target=auto_retrain_loop, daemon=True).start()
load_dotenv()

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="ArchiMind",
    description="AI-driven Architecture Recommendation System",
    version="1.0.0"
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production"),
    max_age=3600  # 1 hour session timeout
)
app.include_router(auth.router)
app.include_router(hybrid_router)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

extractor = SRSExtractor(
    hf_api_key=os.getenv("HF_API_KEY")
)
app.include_router(
    srs_router,
    tags=["Extraction"]
)
# ============================================================
# Templates & Static Files
# ============================================================

templates = Jinja2Templates(directory="presentation/templates")

app.mount(
    "/static",
    StaticFiles(directory="presentation/static"),
    name="static"
)

# ============================================================
# Home Page
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/Admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):

    # 📊 most selected architecture (top 1)
    pipeline = [
        {
            "$match": {
                "selected_architecture": {
                    "$nin": ["Unknown", None, ""]
                }
            }
        },
        {
            "$group": {
                "_id": "$selected_architecture",
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]

    result = list(db.hybrid_method.aggregate(pipeline))

    most_arch = result[0]["_id"] if result else "N/A"
    most_count = result[0]["count"] if result else 0

    # 📈 success based on progress ≥ 70%
    total_projects = db.projects.count_documents({})

    successful_projects = db.projects.count_documents({
        "progress": {"$gte": 70}
    })

    success_rate = 0
    status = "Needs Improvement"

    if total_projects > 0:
        success_rate = int((successful_projects / total_projects) * 100)

        if success_rate >= 70:
            status = "Good"

    # ⏱️ recent projects
    recent_projects = list(
        db.projects.find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(5)
    )

    # 📊 chart data (ALL architectures)
    arch_pipeline = [
        {
            "$match": {
                "selected_architecture": {
                    "$nin": ["Unknown", None, ""]
                }
            }
        },
        {
            "$group": {
                "_id": "$selected_architecture",
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"count": -1}}
    ]

    arch_data = list(db.hybrid_method.aggregate(arch_pipeline))

    arch_labels = [item["_id"] for item in arch_data]
    arch_counts = [item["count"] for item in arch_data]

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "most_arch": most_arch,
            "most_count": most_count,
            "success_rate": success_rate,
            "generated": successful_projects,
            "total_projects": total_projects,
            "status": status,
            "recent_projects": recent_projects,
            "arch_labels": arch_labels,
            "arch_counts": arch_counts
        }
    )
@app.get("/admin/users", response_class=HTMLResponse)
async def get_all_users(request: Request):
    users = list(db.Users.find({}, {"password": 0}))

    enriched_users = []

    # 🔥 function تتحط جوه الفنكشن عادي
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(i) for i in obj]
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

    for user in users:
        user_projects = get_user_projects(str(user["_id"]))

        # 🔥 الحل النهائي
        user_projects = serialize(user_projects)

        user["projects"] = user_projects
        enriched_users.append(user)

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "users": enriched_users
        }
    )


@app.get("/admin/projects", response_class=HTMLResponse)
async def get_all_projects(request: Request):
    projects = list(db.projects.find({}, {"_id": 0}))

    return templates.TemplateResponse(
        "projects.html",
        {
            "request": request,
            "projects": projects
        }
    )


@app.get("/admin/download-adl/{project_id}")
def download_adl(project_id: str):

    doc = db.architecture_reports.find_one({"project_id": project_id})

    if not doc:
        raise HTTPException(status_code=404, detail="ADL not found")

    return StreamingResponse(
        io.BytesIO(doc["report_pdf"]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={project_id}_adl.pdf"
        }
    )




@app.get("/generate/{project_id}")
def generate_architecture(project_id: str):
    print("/generating endpointttt HIT with project_id =", project_id, flush=True)

    # ==========================================================
    # 1. Load selected architecture style
    # ==========================================================
    hybrid_doc = db.hybrid_method.find_one({"project_id": project_id})

    if not hybrid_doc or not hybrid_doc.get("selected_architecture"):
        raise HTTPException(
            status_code=400,
            detail="No selected architecture found for this project"
        )

    selected_architecture = hybrid_doc["selected_architecture"]
    project_doc = db.projects.find_one({"project_id": project_id})

    if not project_doc or not project_doc.get("project_name"):
     raise HTTPException(
        status_code=400,
        detail="Project name not found"
    )

    system_name = project_doc["project_name"]

    # ==========================================================
    # 2. Load input data
    # ==========================================================
    try:
        
        with open("data/outputs/functional_requirements.json", encoding="utf-8") as f:
            functional_requirements = json.load(f)

        with open("data/outputs/non_functional_requirements.json", encoding="utf-8") as f:
            non_functional_requirements = json.load(f)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing input file: {e}")

    # ==========================================================
    # 3. Generate architecture
    # ==========================================================
    arch = ai_generate_architecture(
        system_name,
        functional_requirements,
        non_functional_requirements,
        selected_architecture
    )

    # ==========================================================
    # 4. VERIFICATION (non-blocking — always runs, never stops the pipeline)
    # ==========================================================
    verification_result = {}
    try:
        verification_result = run_verification(arch)
        print(f"[verify] run_verification status={verification_result.get('status')}", flush=True)

        verification_pdf_path = generate_verification_pdf(verification_result)
        print(f"[verify] PDF generated at: {verification_pdf_path}", flush=True)

        with open(verification_pdf_path, "rb") as _vf:
            _vbytes = _vf.read()
        print(f"[verify] PDF bytes read: {len(_vbytes)}", flush=True)

        save_verification_report_pdf(project_id, _vbytes)
        print("[verify] verification report saved to MongoDB successfully", flush=True)
    except Exception as e:
        import traceback
        print(f"[verify] ERROR — verification report NOT saved: {e}", flush=True)
        traceback.print_exc()


    # ==========================================================
    # 5. VALIDATION GATE
    # ==========================================================
    # ==========================================================
# 5. VALIDATION (SUCCESSFUL BUT NOT RETURNED)
# ==========================================================
    validation_result = {}

    try:
      validation_result = run_validation(arch)
      generate_validation_pdf(validation_result)
    except Exception as e:
      print("Validation skipped:", e)
    # ==========================================================
    # 6. Persist architecture outputs
    # ==========================================================
    os.makedirs("data/outputs", exist_ok=True)

    with open("data/outputs/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)

    with open("data/outputs/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_result, f, indent=2)

    with open("data/outputs/architecture.verification.json", "w", encoding="utf-8") as f:
        json.dump(verification_result, f, indent=2)


    acme = convert_to_acme(arch)
    with open("data/outputs/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    # ==========================================================
    # 7. Generate PlantUML views
    # ==========================================================
    uml, data = generate_usecase_ai(functional_requirements, arch["system"])
    if not data or not data.get("actors"):
     data = {"actors": ["User"], "usecases": []}

    with open("data/outputs/dfd_context.puml", "w", encoding="utf-8") as f:
        f.write(convert_to_dfd_context(arch))

    with open("data/outputs/process_view.puml", "w", encoding="utf-8") as f:
        f.write(convert_to_process_view(arch))

    with open("data/outputs/deployment_view.puml", "w", encoding="utf-8") as f:
        f.write(convert_to_deployment_view(arch))

    with open("data/outputs/architecture_c4.puml", "w", encoding="utf-8") as f:
        f.write(convert_to_c4_plantuml(arch))

    

   
    with open("data/outputs/usecase_view.puml", "w", encoding="utf-8") as f:
     f.write(uml) # ==========================================================
    # 8. Render diagrams (PlantUML → PNG)
    # ==========================================================
    PLANTUML_JAR = os.path.join(
        os.path.dirname(__file__),
        "infrastructure",
        "tools",
        "plantuml.jar"
    )

    subprocess.run([
        r"C:\Program Files\Java\jdk-21\bin\java.exe",
        "-jar",
        PLANTUML_JAR,
        "-tpng",
        "data/outputs/dfd_context.puml",
        "data/outputs/process_view.puml",
        "data/outputs/deployment_view.puml",
        "data/outputs/architecture_c4.puml",
        "data/outputs/usecase_view.puml"
    ], check=True)

    # ==========================================================
    # 9. Generate final architecture report
    # ==========================================================
    pdf_path = generate_report(project_id)
    with open(pdf_path, "rb") as f:
     pdf_bytes = f.read()

    save_architecture_report_pdf(project_id, pdf_bytes)
    # ==========================================================
    # 10. Return SUCCESS output only
    # ==========================================================
    return FileResponse(
        path=pdf_path,
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )



@app.get("/download-report")
def download_report():
    return FileResponse(
        path="data/outputs/architecture_report.pdf",
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )



@app.get("/ArchitectureGenerator")
def serve_archgen(request: Request):
    return templates.TemplateResponse("ArchGen.html", {"request": request})

@app.get("/Login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    error: str = None,
    logout: str = None,
    info: str = None
):
    """
    Display login page with optional error message
    """
    error_message = None
    info_message = None
    
    if error == "invalid":
        error_message = "Invalid email or password. Please try again."
    elif error == "server":
        error_message = "Server error occurred. Please try again later."
    elif error == "required":
        error_message = "Please login to access this page."

    if logout == "1":
        info_message = "Thank you for visiting our website!"
    
    if info == "created":
       info_message = "Your account created successfully! Login Now" 
    
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error_message,
            "info": info_message
        }
    )

@app.get("/Dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user_session = request.session.get("user")

    if not user_session:
        return RedirectResponse(
            url="/Login?error=required",
            status_code=303
        )

    user_id = user_session["id"]          # 🔥
    projects = get_user_projects(user_id) # 🔥

    user = {
        "full_name": user_session.get("name", "User"),
        "email": user_session.get("email", ""),
        "role": user_session.get("role", "User")
    }

    return templates.TemplateResponse(
        "Dashboard.html",
        {
            "request": request,
            "user": user,
            "projects": projects   # 🔥 ده اللي كان ناقص
        }
    )

@app.get("/phase4/{project_id}")
def get_phase4(project_id: str):
    return generate_phase4(project_id)


@app.post("/logout")
async def logout(request: Request):
    """
    Clear session and logout user
    """
    request.session.clear()
    return {"status": "success"}



@app.get("/Signup", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse(
        "Signup.html",
        {"request": request}
    )
# ============================================================
# API Routes
# ============================================================



app.include_router(
    architecture_router,
    prefix="/api",
    tags=["Architecture"]
)

app.include_router(
    hybrid_router,
    prefix="/api",        # 👈 API namespace
    tags=["Architecture"]
)

@app.get("/api/report/{project_id}")
def get_report(project_id: str):

    doc = db.architecture_reports.find_one({"project_id": project_id})

    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return StreamingResponse(
        io.BytesIO(doc["report_pdf"]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=architecture_report.pdf"
        }
    )


@app.get("/download-validation-report")
def download_validation_report():
    return FileResponse(
        path="data/outputs/architecture_validation_report.pdf",
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=architecture_validation_report.pdf"
        }
    )


@app.get("/download-verification-report")
def download_verification_report():
    """
    Serve the architecture verification PDF.

    Returns the success report when verification passed, or the
    'problems' report when verification failed. Whichever exists
    on disk is the one returned.
    """
    success_path = "data/outputs/architecture_verification_report.pdf"
    problems_path = "data/outputs/architecture_verification_problems.pdf"

    if os.path.exists(success_path):
        path, filename = success_path, "architecture_verification_report.pdf"
    elif os.path.exists(problems_path):
        path, filename = problems_path, "architecture_verification_problems.pdf"
    else:
        raise HTTPException(
            status_code=404,
            detail="Verification report not found. Run verification first."
        )

    return FileResponse(
        path=path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={filename}"
        }
    )