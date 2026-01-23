from fastapi import APIRouter, UploadFile, File
import os
from application.extraction.srs_extractor import SRSExtractor
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

extractor = SRSExtractor(
    hf_api_key=os.getenv("HF_API_KEY")
)

@router.post("/extract")
async def extract_srs(file: UploadFile = File(...)):
    pdf_path = os.path.join(UPLOAD_DIR, "srs.pdf")

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # ✅ PDF extraction من application layer
    text = extractor.extract_text_from_pdf(pdf_path)

    # ✅ Requirement extraction
    result = extractor.extract_requirements(text)

    return result
