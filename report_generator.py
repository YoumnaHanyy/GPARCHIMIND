import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).parent

input_req = BASE_DIR / "input" / "requirements.json"
acme_file = BASE_DIR / "output" / "architecture.acme"
pdf_file = BASE_DIR / "output" / "architecture_report.pdf"

c = canvas.Canvas(str(pdf_file), pagesize=A4)
width, height = A4

y = height - 50

def write_line(text):
    global y
    if y < 50:
        c.showPage()
        y = height - 50
    c.drawString(50, y, text)
    y -= 15

# Title
write_line("ARCHITECTURE REPORT")
write_line("=" * 80)
y -= 20

# 1. Requirements
write_line("1. REQUIREMENTS")
write_line("-" * 80)

with open(input_req, "r", encoding="utf-8") as f:
    reqs = json.dumps(json.load(f), indent=2)

for line in reqs.split("\n"):
    write_line(line)

y -= 20

# 2. ACME ADL
write_line("2. ARCHITECTURE (ACME ADL)")
write_line("-" * 80)

with open(acme_file, "r", encoding="utf-8") as f:
    for line in f:
        write_line(line.rstrip())

c.save()

print("📄 Architecture PDF report generated successfully!")
