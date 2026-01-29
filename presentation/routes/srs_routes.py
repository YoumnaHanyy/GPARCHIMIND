from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import re
import uuid

import fitz  # PyMuPDF

from application.extraction.extraction_service import process_srs
from ai.inference.predict_type_level import predict_and_save_nfr

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# 1) Helper to remove ObjectId fields (Mongo style)
# ============================================================
def clean_object_id(items: list):
    cleaned = []
    for item in items:
        item = dict(item)
        item.pop("_id", None)
        item.pop("project_id", None)
        cleaned.append(item)
    return cleaned


# ============================================================
# 2) PDF text extraction
# ============================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    try:
        for page in doc:
            parts.append(page.get_text("text"))
    finally:
        doc.close()
    return "\n".join(parts)


# ============================================================
# 3) NEW V&V Gate: Keyword / Heading-based validation
#    - Based only on keywords you asked for:
#      "functional requirement", "non-functional requirement", "fr", "nfr"
# ============================================================

# Strong heading keywords (most reliable)
HEADING_PATTERNS = [
    r"\bfunctional\s+requirements?\b",
    r"\bnon[\-\s]?functional\s+requirements?\b",
]

# FR / NFR abbreviations (acceptable but weaker alone)
ABBREV_PATTERNS = [
    r"\bfr\b",
    r"\bnfr\b",
]

# compile regex
HEADING_RE = [re.compile(p, re.IGNORECASE) for p in HEADING_PATTERNS]
ABBREV_RE  = [re.compile(p, re.IGNORECASE) for p in ABBREV_PATTERNS]


def validate_srs_by_keywords(text: str):
    """
    PASS if:
      - any heading keyword is found (functional requirements / non-functional requirements)
        OR
      - both abbreviations FR and NFR are found somewhere in the document
    """

    # Work on a "front window" (titles/sections are usually early)
    # But still search full text as a backup
    front = text[:12000] if text else ""
    low_full = text.lower() if text else ""
    low_front = front.lower() if front else ""

    # Scan lines (gives you good samples to show in UI/log)
    lines = [ln.strip() for ln in front.splitlines() if ln.strip()]
    lines_to_scan = lines[:200]  # first ~200 lines enough for headings

    heading_hits = []
    abbrev_hits = {"fr": 0, "nfr": 0}

    # 1) Heading hits in the first lines
    for ln in lines_to_scan:
        for rx in HEADING_RE:
            if rx.search(ln):
                heading_hits.append(ln)

    # 2) Abbreviation hits in full text (FR / NFR)
    # We count on FULL text because FR/NFR could appear later
    for rx in ABBREV_RE:
        m = rx.findall(low_full)
        if rx.pattern.lower().find(r"\bfr\b") != -1:
            abbrev_hits["fr"] = len(m)
        if rx.pattern.lower().find(r"\bnfr\b") != -1:
            abbrev_hits["nfr"] = len(m)

    # Decide PASS/FAIL
    found_heading = len(heading_hits) > 0
    found_both_abbrev = (abbrev_hits["fr"] > 0 and abbrev_hits["nfr"] > 0)

    passed = found_heading or found_both_abbrev

    return {
        "passed": passed,
        "method": "keyword_heading_validation",
        "stats": {
            "text_len": len(text) if text else 0,
            "heading_hits_count": len(heading_hits),
            "fr_count": abbrev_hits["fr"],
            "nfr_count": abbrev_hits["nfr"],
            "front_window_chars": len(front),
        },
        "samples": {
            "heading_hits": heading_hits[:8],
        },
        "decision": {
            "found_heading_keywords": found_heading,
            "found_both_fr_nfr": found_both_abbrev
        }
    }


# ============================================================
# 4) API Endpoint: /extract
#    - Save file
#    - V&V Gate (NEW)
#    - If PASS -> process_srs + predict NFR
# ============================================================
@router.post("/extract")
async def extract_srs(file: UploadFile = File(...)):
    try:
        project_id = 2  # temporary

        # 1) Save uploaded file with unique name
        safe_name = f"srs_{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(UPLOAD_DIR, safe_name)

        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)

        # 2) NEW V&V: validate document contains the required keywords
        text = extract_text_from_pdf(pdf_path)
        vv = validate_srs_by_keywords(text)

        if not vv["passed"]:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "FAIL",
                    "stage": "V&V",
                    "message": (
                        "Invalid SRS: No clear keywords were detected "
                        "(Functional Requirements / Non-Functional Requirements / System Requirements / FR / NFR). "
                        "Please upload an SRS that contains these sections or labels."
                    ),
                    "vv": vv
                }
            )

        # 3) If PASS: Extract FR + NFR
        extraction_result = process_srs(
            pdf_path=pdf_path,
            project_id=project_id,
            hf_key=os.getenv("HF_API_KEY")
        )

        # 4) Predict NFR types
        predictions = predict_and_save_nfr()

        # 5) Return to UI
        return {
            "status": "PASS",
            "stage": "Extraction",
            "message": "Valid SRS detected. Requirements extraction completed successfully.",
            "vv": vv,
            "functional": clean_object_id(extraction_result.get("functional", [])),
            "nfr_predictions": clean_object_id(predictions)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "error": str(e)}
        )
