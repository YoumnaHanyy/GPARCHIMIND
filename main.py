import json
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from report_generator import generate_report
from fastapi.templating import Jinja2Templates
from adl.json_to_c4_plantuml import convert_to_c4_plantuml
from adl.adl_generator import generate_adl
from adl.json_to_acme import convert_to_acme


app = FastAPI()
templates = Jinja2Templates(directory="templates")




# ---------------- UI ----------------
@app.get("/")
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/generate")
def generate_architecture():

    with open("input/requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    adl, validation = generate_adl(requirements)

    with open("output/architecture.adl.json", "w", encoding="utf-8") as f:
        json.dump(adl, f, indent=2)

    with open("output/architecture.validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    acme = convert_to_acme(adl)
    with open("output/architecture.acme", "w", encoding="utf-8") as f:
        f.write(acme)

    # ---- C4 PlantUML ----
    c4_puml = convert_to_c4_plantuml(adl)
    with open("output/architecture_c4.puml", "w", encoding="utf-8") as f:
        f.write(c4_puml)

    subprocess.run([
        r"C:\Program Files\Java\jdk-21\bin\java.exe",
        "-jar",
        "plantuml.jar",
        "-tpng",
        "output/architecture_c4.puml"
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
