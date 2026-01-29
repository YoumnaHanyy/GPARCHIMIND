from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uuid
import traceback

from application.extraction.extraction_service import process_srs
from ai.inference.predict_type_level import predict_and_save_nfr
from application.extraction.ordinal_service import execute_ordinal_method
from application.extraction.binary_service import execute_binary_method
from application.extraction.weighted_service import execute_weighted_method
from application.extraction.nfr_stats_service import compute_nfr_statistics
from application.extraction.functional_service import execute_functional_method
from application.extraction.hybrid_service import execute_hybrid_method


from infrastructure.repositories.weighted_repository import save_weighted_result
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
        project_id = f"proj_{uuid.uuid4().hex[:8]}"

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

        # functional architecture (🆕)
        functional_result = execute_functional_method(project_id)


        # 3️⃣ predict NFR type + level
        predictions = predict_and_save_nfr()

        # 🔹 NEW: compute stats for weighted
        freq_norm, must_norm, importance = compute_nfr_statistics(predictions)



        # 4️⃣ ordinal
        ordinal_result = execute_ordinal_method()

        # 5️⃣ 🔥 binary (ده كان ناقص)
        binary_result = execute_binary_method()

        # 6️⃣ weighted (🆕)
        weighted_result = execute_weighted_method(
                                     freq_norm=freq_norm,
                                     must_norm=must_norm,
                                     importance=importance
                                    )
        
        hybrid_result = execute_hybrid_method(
                        project_id,
                        functional_result,
                        ordinal_result,
                        binary_result,
                        weighted_result
        )

        save_weighted_result(project_id, weighted_result)
        # 6️⃣ response للـ UI
        return {
            "project_id": project_id, 
            "functional": clean_object_id(extraction_result["functional"]),
            "nfr_predictions": clean_object_id(predictions),
            "functional_method": functional_result,
            "ordinal_method": ordinal_result["result"],
            "binary_method": binary_result,
            "weighted_method": weighted_result,
               "hybrid_method": hybrid_result
        }

    except Exception as e:
       tb = traceback.extract_tb(e.__traceback__)
       last_trace = tb[-1]  # آخر مكان وقع فيه الغلط

       return JSONResponse(
          status_code=500,
          content={
            "error": str(e),
            "file": os.path.relpath(last_trace.filename),
            "line": last_trace.lineno,
            "code": last_trace.line
          }
        )
