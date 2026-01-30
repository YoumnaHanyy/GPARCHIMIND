from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io
from infrastructure.database import db
from infrastructure.repositories.ADL_repository import save_architecture_report_pdf
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
from ai.json_to_context_view import convert_to_context_view
from application.extraction.adl.json_to_acme import convert_to_acme
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import os
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
import json
# Import API routes
from fastapi.responses import HTMLResponse
from presentation.routes.architecture_routes import router as architecture_router
from presentation.routes.srs_routes import router as srs_router
from dotenv import load_dotenv
import os

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



@app.get("/generate/{project_id}")
def generate_architecture(project_id: str):

    hybrid_doc = db.hybrid_method.find_one({"project_id": project_id})

    if not hybrid_doc or not hybrid_doc.get("selected_architecture"):
        raise HTTPException(
            status_code=400,
            detail="No selected architecture found for this project"
        )
    selected_architecture = hybrid_doc["selected_architecture"]




    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parents[0]  # لو main.py في root

    # -------- load system info --------
    with open("data/outputs/input/requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    # -------- load FRs --------
    with open("data/outputs/functional_requirements.json", "r", encoding="utf-8") as f:
        functional_requirements = json.load(f)

    # -------- load NFRs --------
    with open("data/outputs/non_functional_requirements.json", "r", encoding="utf-8") as f:
        non_functional_requirements = json.load(f)

    # -------- generate architecture --------
    arch = ai_generate_architecture(
        requirements["system_name"],
        functional_requirements,
        non_functional_requirements,
        selected_architecture
    )

    with open("data/outputs/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)


    with open("data/outputs/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(arch["critique"], f, indent=2)


    acme = convert_to_acme(arch)
    with open("data/outputs/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    # ---- C4 PlantUML ----
    c4_puml = convert_to_c4_plantuml(arch)
    with open("data/outputs/architecture_c4.puml", "w", encoding="utf-8") as f:
        f.write(c4_puml)

    # ---- Context View ----
    context_puml = convert_to_context_view(arch)

    with open("data/outputs/context_view.puml", "w", encoding="utf-8") as f:
       f.write(context_puml)

    # ---- DFD Context View (Level 0) ----
    dfd_puml = convert_to_dfd_context(arch)

    with open("data/outputs/dfd_context.puml", "w", encoding="utf-8") as f:
       f.write(dfd_puml)

    # ---- Process View ----
    process_puml = convert_to_process_view(arch)

    with open("data/outputs/process_view.puml", "w", encoding="utf-8") as f:
       f.write(process_puml)



    PLANTUML_JAR = os.path.join(
    os.path.dirname(__file__),
    "infrastructure",
    "tools",
    "plantuml.jar"
    )


    # ---- Physical View (Deployment Diagram) ----
    deployment_puml = convert_to_deployment_view(arch)

    with open("data/outputs/deployment_view.puml", "w", encoding="utf-8") as f:
      f.write(deployment_puml)
    subprocess.run([
    r"C:\Program Files\Java\jdk-21\bin\java.exe",
    "-jar",
    PLANTUML_JAR,
    "-tpng",
    "data/outputs/architecture_c4.puml",
    "data/outputs/dfd_context.puml",
    "data/outputs/context_view.puml",
    "data/outputs/process_view.puml",
    "data/outputs/deployment_view.puml"
], check=True)




    # ---- Generate PDF automatically ----
    pdf_path = generate_report(project_id)
    
    with open(pdf_path, "rb") as f:
     pdf_bytes = f.read()

    save_architecture_report_pdf(
    project_id=project_id,
    pdf_bytes=pdf_bytes
     )
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
    logout: str = None
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
