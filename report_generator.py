import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

input_req = BASE_DIR / "input" / "requirements.json"
adl_json = BASE_DIR / "output" / "architecture.adl.json"
acme_file = BASE_DIR / "output" / "architecture.acme"
report_file = BASE_DIR / "output" / "architecture_report.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("=================================\n")
    report.write("ARCHITECTURE REPORT\n")
    report.write("=================================\n\n")

    # 1. Requirements
    report.write("1. REQUIREMENTS\n")
    report.write("---------------------------------\n")
    with open(input_req, "r", encoding="utf-8") as f:
        report.write(json.dumps(json.load(f), indent=2))
    report.write("\n\n")

     # 3. Architecture ACME
    report.write("3. ARCHITECTURE (ACME ADL)\n")
    report.write("---------------------------------\n")
    with open(acme_file, "r", encoding="utf-8") as f:
        report.write(f.read())

print("📄 Architecture report generated successfully!")
