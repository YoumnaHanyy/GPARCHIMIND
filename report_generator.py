def generate_report():
    import json
    from pathlib import Path
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    BASE_DIR = Path(__file__).parent

    input_req = BASE_DIR / "input" / "requirements.json"
    acme_file = BASE_DIR / "output" / "architecture.acme"
    diagram_file = BASE_DIR / "output" / "architecture_c4.png"
    context_diagram = BASE_DIR / "output" / "context_view.png"
    dfd_context_diagram = BASE_DIR / "output" / "dfd_context.png"
    pdf_file = BASE_DIR / "output" / "architecture_report.pdf"



    c = canvas.Canvas(str(pdf_file), pagesize=A4)
    width, height = A4

    y = height - 50

    def write_line(text):
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, text)
        y -= 15

    write_line("ARCHITECTURE REPORT")
    write_line("=" * 80)
    y -= 20

    write_line("1. REQUIREMENTS")
    write_line("-" * 80)

    with open(input_req, "r", encoding="utf-8") as f:
        reqs = json.dumps(json.load(f), indent=2)

    for line in reqs.split("\n"):
        write_line(line)

    y -= 30

    write_line("2. ARCHITECTURE (ACME ADL)")
    write_line("-" * 80)

    with open(acme_file, "r", encoding="utf-8") as f:
        for line in f:
            write_line(line.rstrip())




    y -= 30
    write_line("3. CONTEXT VIEW")
    write_line("-" * 80)

    if context_diagram.exists():
      if y < 300:
          c.showPage()
          y = height - 50


    c.drawImage(
           str(context_diagram),
           50,
           y - 250,
           width=500,
           height=250,
           preserveAspectRatio=True,
           mask='auto'
        )
    y -= 270

    y -= 30
    write_line("3.1 DATA CONTEXT VIEW (LEVEL 0 DFD)")
    write_line("-" * 80)

    if dfd_context_diagram.exists():
     if y < 300:
        c.showPage()
        y = height - 50

    c.drawImage(
        str(dfd_context_diagram),
        50,
        y - 250,
        width=500,
        height=250,
        preserveAspectRatio=True,
        mask='auto'
    )
    y -= 270
    

    y -= 30
    write_line("4. C4 CONTAINER DIAGRAM")
    write_line("-" * 80)

    if diagram_file.exists():
        if y < 300:
            c.showPage()
            y = height - 50

        c.drawImage(
            str(diagram_file),
            50,
            y - 250,
            width=500,
            height=250,
            preserveAspectRatio=True,
            mask='auto'
        )
        y -= 270
    else:
        write_line("C4 diagram image not found.")

    c.save()
    return pdf_file
