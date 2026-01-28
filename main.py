from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ai.json_to_process_view import convert_to_process_view
from fastapi.responses import FileResponse
from ai.json_to_dfd_context import convert_to_dfd_context
from application.extraction.reporting.report_generator import generate_report
from fastapi.templating import Jinja2Templates
from ai.json_to_deployment_view import convert_to_deployment_view

from ai.json_to_c4_plantuml import convert_to_c4_plantuml
from ai.ai_engine import ai_generate_architecture
from ai.json_to_context_view import convert_to_context_view
from application.extraction.adl.json_to_acme import convert_to_acme
import os
import subprocess
import json
from application.extraction.srs_extractor import SRSExtractor
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



@app.get("/generate")
def generate_architecture():

    with open("data/outputs/input/requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    arch = ai_generate_architecture(
    requirements["system_name"],
    requirements["functional_requirements"],
    requirements["non_functional_requirements"],
    requirements["architecture_style"]
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
    pdf_path = generate_report()

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
def login(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@app.get("/Dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )


@app.get("/Signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse(
        "Signup.html",
        {"request": request}
    )
# ============================================================
# API Routes
# ============================================================

app.include_router(
    architecture_router,
    prefix="/api",        # 👈 API namespace
    tags=["Architecture"]
)
