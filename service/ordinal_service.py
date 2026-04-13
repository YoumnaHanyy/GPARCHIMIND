import json
from ai.methods.ordinal_method import run_ordinal_method
from infrastructure.repositories.ordinal_repository import save_ordinal_result
from infrastructure.repositories.architecture_repository import get_architecture_dataset

PREDICTIONS_PATH = "data/outputs/nfr_predictions_type_level.json"

def execute_ordinal_method(project_id):
    # 1️⃣ load predictions (generated earlier)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # 2️⃣ load architecture dataset from MongoDB
    architecture_data = get_architecture_dataset()

    # 3️⃣ run ordinal logic
    result = run_ordinal_method(predictions, architecture_data)

    # 4️⃣ save result to MongoDB
    saved_id = save_ordinal_result(
        project_id,
       {
        "method": "ordinal",
        "result": result
       }
    )

    return {
        "id": str(saved_id),
        "result": result
    }
