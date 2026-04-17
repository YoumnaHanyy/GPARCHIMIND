from fastapi import APIRouter
from pydantic import BaseModel
from infrastructure.repositories.hybrid_repository import save_hybrid_result
# 1. استدعي الدالة من المسار بتاعها
from ai.inference.feature_extractor import generate_phase4 

router = APIRouter()

class SaveHybridRequest(BaseModel):
    project_id: str
    selected_architecture: str
    hybrid_result: list

@router.post("/save-hybrid-result")
def save_hybrid_result_route(req: SaveHybridRequest):
    # 2. حفظ الاختيار في قاعدة البيانات
    save_hybrid_result(
        project_id=req.project_id,
        result=req.hybrid_result,
        selected_architecture=req.selected_architecture
    )
    
    # 3. 🔥 اللحظة الحاسمة: تشغيل المرحلة الرابعة بناءً على الاختيار الجديد
    # دلوقتي الدالة لما تشتغل هتروح تلاقي selected_architecture متسيف فعلاً في الـ DB
    phase4_data = generate_phase4(req.project_id)
    
    return {
        "message": "Hybrid result saved successfully",
        "phase4": phase4_data  # 4. ابعتي البيانات الجديدة للـ Frontend عشان يحدث الـ UI
    }