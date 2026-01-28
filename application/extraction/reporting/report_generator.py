def generate_report():
    import json
    from pathlib import Path
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    BASE_DIR = Path(__file__).resolve()
    PROJECT_ROOT = BASE_DIR.parents[3]


    requirements_file = PROJECT_ROOT / "data" / "outputs" / "input" / "requirements.json"
    frs_file = PROJECT_ROOT / "data" / "outputs" / "functional_requirements.json"
    nfrs_file = PROJECT_ROOT / "data" / "outputs" / "non_functional_requirements.json"

    acme_file = PROJECT_ROOT / "data" / "outputs" / "architecture.acme"

    context_diagram = PROJECT_ROOT / "data" / "outputs" / "context_view.png"
    dfd_context_diagram = PROJECT_ROOT / "data" / "outputs" / "dfd_context.png"
    container_diagram = PROJECT_ROOT / "data" / "outputs" / "architecture_c4.png"
    process_diagram = PROJECT_ROOT / "data" / "outputs" / "process_view.png"
    deployment_diagram = PROJECT_ROOT / "data" / "outputs" / "deployment_view.png"

    pdf_file = PROJECT_ROOT / "data" / "outputs" / "architecture_report.pdf"

    c = canvas.Canvas(str(pdf_file), pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    # ---------- Helpers ----------
    def new_page():
        nonlocal y
        c.showPage()
        y = height - 2 * cm

    def title(text):
        nonlocal y
        if y < 3 * cm:
            new_page()
        c.setFont("Helvetica-Bold", 20)
        c.drawString(2 * cm, y, text)
        y -= 1.2 * cm

    def section(text):
        nonlocal y
        if y < 3 * cm:
            new_page()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y, text)
        y -= 0.8 * cm
        c.line(2 * cm, y, width - 2 * cm, y)
        y -= 0.8 * cm

    def paragraph(text):
        nonlocal y
        c.setFont("Helvetica", 11)
        for line in text.split("\n"):
            if y < 2 * cm:
                new_page()
            c.drawString(2 * cm, y, line)
            y -= 0.6 * cm
        y -= 0.4 * cm

    def bullet_list(items):
        nonlocal y
        c.setFont("Helvetica", 11)
        for item in items:
            if y < 2 * cm:
                new_page()
            c.drawString(2.4 * cm, y, f"- {item}")
            y -= 0.6 * cm
        y -= 0.4 * cm

    def image_section(title_text, image_path):
        nonlocal y
        section(title_text)
        if image_path.exists():
            if y < 8 * cm:
                new_page()
            c.drawImage(
                str(image_path),
                2 * cm,
                y - 7 * cm,
                width=width - 4 * cm,
                height=7 * cm,
                preserveAspectRatio=True,
                mask="auto"
            )
            y -= 7.8 * cm
        else:
            paragraph("Diagram not available.")

    # ---------- Title ----------
    title("Architecture Design Report")

    paragraph(
        "This document presents the architectural design of the system, "
        "including its context, logical structure, runtime behavior, "
        "and physical deployment."
    )

    # ---------- Requirements ----------
    section("1. System Requirements")

    # ---- Load system info ----
    with open(requirements_file, "r", encoding="utf-8") as f:
      req = json.load(f)

    system_name = req.get("system_name")
    architecture_style = req.get("architecture_style")

# ---- Load FRs ----
    with open(frs_file, "r", encoding="utf-8") as f:
      functional_requirements = json.load(f)

# ---- Load NFRs ----
    with open(nfrs_file, "r", encoding="utf-8") as f:
     non_functional_requirements = json.load(f)

    paragraph(f"System Name: {system_name}")
    paragraph(f"Architecture Style: {architecture_style}")

    section("Functional Requirements")
    bullet_list([fr["title"] for fr in functional_requirements])

    section("Non-Functional Requirements")

    for nfr in non_functional_requirements:
    # Title as bullet
     paragraph(f"- {nfr['title']}")
     paragraph(f"      Description: {nfr['description']}")






    # ---------- ACME ----------
    section("2. Formal Architecture Specification (ACME ADL)")
    paragraph(
        "The following section provides a formal architectural specification "
        "using ACME ADL, capturing components, connectors, and architectural properties."
    )

    with open(acme_file, "r", encoding="utf-8") as f:
        c.setFont("Courier", 9)
        for line in f:
            if y < 2 * cm:
                new_page()
                c.setFont("Courier", 9)
            c.drawString(2 * cm, y, line.rstrip())
            y -= 0.45 * cm
        y -= 0.6 * cm

    # ---------- Views ----------
    image_section("3. Context View", context_diagram)
    image_section("3.1 Data Context View (Level 0 DFD)", dfd_context_diagram)
    image_section("4. Logical View (C4 Container Diagram)", container_diagram)
    image_section("5. Process View (Runtime Interaction)", process_diagram)
    image_section("6. Physical View (Deployment Diagram)", deployment_diagram)

    c.save()
    return pdf_file
