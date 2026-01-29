from fastapi import APIRouter
from pydantic import BaseModel
from infrastructure.repositories.hybrid_repository import save_hybrid_result

router = APIRouter()

class SaveHybridRequest(BaseModel):
    project_id: str
    selected_architecture: str
    hybrid_result: list

@router.post("/save-hybrid-result")
def save_hybrid_result_route(req: SaveHybridRequest):
    save_hybrid_result(
        project_id=req.project_id,
        result=req.hybrid_result,
        selected_architecture=req.selected_architecture
    )
    return {"message": "Hybrid result saved successfully"}
