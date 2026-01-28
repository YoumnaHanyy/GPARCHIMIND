import json
import subprocess
import zipfile
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from adl.json_to_process_view import convert_to_process_view
from adl.json_to_dfd_context import convert_to_dfd_context
from adl.json_to_deployment_view import convert_to_deployment_view
from adl.json_to_c4_plantuml import convert_to_c4_plantuml
from adl.json_to_context_view import convert_to_context_view
from adl.json_to_acme import convert_to_acme

from adl.ai_engine import ai_generate_architecture
from adl.validation.validation_report_generator import generate_validation_pdf
from adl.validation.runner import run_validation
from report_generator import generate_report


app = FastAPI()
templates = Jinja2Templates(directory="templates")


# ---------------- UI ----------------
@app.get("/ArchitectureGenerator")
def serve_archgen(request: Request):
    return templates.TemplateResponse("ArchGen.html", {"request": request})

@app.get("/Register")
def serve_register(request: Request):
    return templates.TemplateResponse("Signup.html", {"request": request})

@app.get("/Login")
def serve_login(request: Request):
    return templates.TemplateResponse("Login.html", {"request": request})

@app.get("/")
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ================= GENERATE =================
@app.get("/generate")
def generate_architecture():

    with open("input/requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    arch = ai_generate_architecture(
        requirements["system_name"],
        requirements["functional_requirements"],
        requirements["non_functional_requirements"],
        requirements["architecture_style"]
    )

    # -------- VALIDATION (always generated) --------
    validation_result = run_validation(arch)
    validation_pdf_path = generate_validation_pdf(validation_result)

    # -------- FAIL: return validation report only --------
    if validation_result["status"] == "FAILED":
        return FileResponse(
            path=validation_pdf_path,
            filename="architecture_validation_problems.pdf",
            media_type="application/pdf"
        )

    # -------- SUCCESS: old logic untouched --------
    with open("output/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)

    with open("output/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(arch["critique"], f, indent=2)

    acme = convert_to_acme(arch)
    with open("output/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    c4_puml = convert_to_c4_plantuml(arch)
    with open("output/architecture_c4.puml", "w", encoding="utf-8") as f:
        f.write(c4_puml)

    context_puml = convert_to_context_view(arch)
    with open("output/context_view.puml", "w", encoding="utf-8") as f:
        f.write(context_puml)

    dfd_puml = convert_to_dfd_context(arch)
    with open("output/dfd_context.puml", "w", encoding="utf-8") as f:
        f.write(dfd_puml)

    process_puml = convert_to_process_view(arch)
    with open("output/process_view.puml", "w", encoding="utf-8") as f:
        f.write(process_puml)

    deployment_puml = convert_to_deployment_view(arch)
    with open("output/deployment_view.puml", "w", encoding="utf-8") as f:
        f.write(deployment_puml)

    subprocess.run([
        r"C:\Program Files\Java\jdk-24\bin\java.exe",
        "-jar",
        "plantuml.jar",
        "-tpng",
        "output/architecture_c4.puml",
        "output/dfd_context.puml",
        "output/context_view.puml",
        "output/process_view.puml",
        "output/deployment_view.puml"
    ], check=True)

    # ---- Generate PDF automatically ----
    pdf_path = generate_report()

    zip_path = "output/architecture_outputs.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(
            pdf_path,
            arcname="architecture_report.pdf"
        )
        zipf.write(
            "output/architecture_validation_report.pdf",
            arcname="architecture_validation_report.pdf"
        )

    return FileResponse(
        path=zip_path,
        filename="architecture_outputs.zip",
        media_type="application/zip"
    )


# ---------------- Download PDF Report ----------------
@app.get("/download-report")
def download_report():
    return FileResponse(
        path="output/architecture_report.pdf",
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )

@app.get("/download-validation-report")
def download_validation_report():
    return FileResponse(
        path="output/architecture_validation_report.pdf",
        filename="architecture_validation_report.pdf",
        media_type="application/pdf"
    )
