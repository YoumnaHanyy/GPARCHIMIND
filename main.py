import json

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from adl.adl_generator import generate_adl
from adl.json_to_acme import convert_to_acme

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- Generate Architecture ----------------
@app.post("/generate")
def generate_architecture():

    # Load requirements
    with open("input/requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    # Generate ADL + Validation
    adl, validation = generate_adl(requirements)

    # Save ADL
    with open("output/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(adl, f, indent=2)

    # Save Validation
    with open("output/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    # Convert to ACME
    acme = convert_to_acme(adl)
    with open("output/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    return {
        "status": "Architecture generated successfully",
        "is_valid": validation.get("is_valid", True)
    }

# ---------------- Download PDF Report ----------------
@app.get("/download-report")
def download_report():
    return FileResponse(
        path="output/architecture_report.pdf",
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )
