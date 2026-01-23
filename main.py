import json
from adl.adl_generator import generate_adl
from adl.json_to_acme import convert_to_acme

# ---------------- Load requirements ----------------
with open("input/requirements.json") as f:
    requirements = json.load(f)

# ---------------- Generate ADL + Validation ----------------
adl, validation = generate_adl(requirements)

# ---------------- Save ADL ----------------
with open("output/architecture.adl.json", "w") as f:
    json.dump(adl, f, indent=2)

# ---------------- Save Validation ----------------
with open("output/architecture.validation.json", "w") as f:
    json.dump(validation, f, indent=2)

# ---------------- Convert to ACME ----------------
acme = convert_to_acme(adl)

with open("output/architecture.acme", "w") as f:
    f.write(acme)

# ---------------- Console Output ----------------
print("✅ AI-driven Architecture ADL generated")
print("📊 Architecture metrics computed")
print("🧪 Architecture validation generated")

if not validation.get("is_valid", True):
    print("❌ Architecture is NOT valid – check architecture.validation.json")
else:
    print("✅ Architecture is VALID and production-ready")
