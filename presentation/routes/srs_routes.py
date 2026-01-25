from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os

from application.extraction.extraction_service import process_srs
from ai.inference.predict_type_level import predict_and_save_nfr
from application.extraction.ordinal_service import execute_ordinal_method

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def clean_object_id(items: list):
    cleaned = []
    for item in items:
        item = dict(item)
        item.pop("_id", None)
        item.pop("project_id", None)
        cleaned.append(item)
    return cleaned


@router.post("/extract")
async def extract_srs(file: UploadFile = File(...)):
    try:
        project_id = 2  # مؤقت

        # 1️⃣ save pdf
        pdf_path = os.path.join(UPLOAD_DIR, "srs.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # 2️⃣ extract FR + NFR
        extraction_result = process_srs(
            pdf_path=pdf_path,
            project_id=project_id,
            hf_key=os.getenv("HF_API_KEY")
        )

        # 3️⃣ predict NFR type + level
        predictions = predict_and_save_nfr()

        # 4️⃣ 🔥 run ordinal automatically
        ordinal_result = execute_ordinal_method()

        # 5️⃣ response للـ UI
        return {
            "functional": clean_object_id(extraction_result["functional"]),
            "nfr_predictions": clean_object_id(predictions),
            "ordinal_method": ordinal_result["result"]  # 👈 يظهر فورًا
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
