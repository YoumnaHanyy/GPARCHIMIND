from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
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
