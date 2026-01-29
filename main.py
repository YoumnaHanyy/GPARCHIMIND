from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import os
import json
from service.srs_extractor import SRSExtractor
from presentation.routes.architecture_routes import router as architecture_router
from presentation.routes.srs_routes import router as srs_router
from presentation.routes import auth
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from infrastructure.repositories.project_repo import get_user_projects

load_dotenv()

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="ArchiMind",
    description="AI-driven Architecture Recommendation System",
    version="1.0.0"
)

# ✅ Add session middleware FIRST
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "super-secret-key-change-this-in-production"),
    max_age=3600  # 1 hour session timeout
)

# ✅ Include auth router
app.include_router(auth.router)

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
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


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