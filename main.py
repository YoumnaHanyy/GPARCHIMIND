import json
import subprocess
from fastapi import FastAPI, Request
from adl.json_to_process_view import convert_to_process_view
from fastapi.responses import FileResponse
from adl.json_to_dfd_context import convert_to_dfd_context
from report_generator import generate_report
from fastapi.templating import Jinja2Templates
from adl.json_to_deployment_view import convert_to_deployment_view

from adl.json_to_c4_plantuml import convert_to_c4_plantuml
from adl.ai_engine import ai_generate_architecture
from adl.json_to_context_view import convert_to_context_view
from adl.json_to_acme import convert_to_acme


app = FastAPI()
templates = Jinja2Templates(directory="templates")


# ---------------- UI ----------------
@app.get("/ArchitectureGenerator")
def serve_archgen(request: Request):
    return templates.TemplateResponse("ArchGen.html", {"request": request})

@app.get("/")
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ================= AI ENGINE WRAPPERS =================
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

    with open("output/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)


    with open("output/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(arch["critique"], f, indent=2)


    acme = convert_to_acme(arch)
    with open("output/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    # ---- C4 PlantUML ----
    c4_puml = convert_to_c4_plantuml(arch)
    with open("output/architecture_c4.puml", "w", encoding="utf-8") as f:
        f.write(c4_puml)

    # ---- Context View ----
    context_puml = convert_to_context_view(arch)

    with open("output/context_view.puml", "w", encoding="utf-8") as f:
       f.write(context_puml)

    # ---- DFD Context View (Level 0) ----
    dfd_puml = convert_to_dfd_context(arch)

    with open("output/dfd_context.puml", "w", encoding="utf-8") as f:
       f.write(dfd_puml)

    # ---- Process View ----
    process_puml = convert_to_process_view(arch)

    with open("output/process_view.puml", "w", encoding="utf-8") as f:
       f.write(process_puml)


    # ---- Physical View (Deployment Diagram) ----
    deployment_puml = convert_to_deployment_view(arch)

    with open("output/deployment_view.puml", "w", encoding="utf-8") as f:
      f.write(deployment_puml)
    subprocess.run([
    r"C:\Program Files\Java\jdk-21\bin\java.exe",
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

    return FileResponse(
        path=pdf_path,
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )

# ---------------- Download PDF Report ----------------
@app.get("/download-report")
def download_report():
    return FileResponse(
        path="output/architecture_report.pdf",
        filename="architecture_report.pdf",
        media_type="application/pdf"
    )
