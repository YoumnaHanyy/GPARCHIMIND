from fastapi import APIRouter, UploadFile, File
from infrastructure.database import db

router = APIRouter(prefix="/architecture", tags=["Architecture"])


@router.get("/test-db")
def test_db():
    collections = db.list_collection_names()
    return {
        "status": "connected",
        "collections": collections
    }


# 🔹 endpoint وهمي (stub) عشان الـ UI
@router.post("/upload_srs/")
async def upload_srs_stub(file: UploadFile = File(...)):
    return {
        "functional": ["Sample FR 1", "Sample FR 2"],
        "non_functional": ["Performance", "Security"],
        "architecture_recommendations": {
            "functional_method": {
                "chosen_architectures": ["Layered"],
                "explanation": {
                    "Layered": "Good separation of concerns"
                }
            },
            "ordinal_method": ["Microservices", "Layered"],
            "binary_method": ["Client-Server"],
            "weighted_score_method": {
                "Layered": 0.82,
                "Microservices": 0.76
            },
            "hybrid_method": ["Layered"]
        }
    }
