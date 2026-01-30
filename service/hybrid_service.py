from infrastructure.database import db
from ai.methods.hybrid_method import hybrid_aggregation 
from infrastructure.repositories.hybrid_repository import save_hybrid_result
def execute_hybrid_method(project_id, functional, ordinal, binary, weighted):

    # ===============================
    # 1️⃣ Normalize ORDINAL
    # ===============================
    if isinstance(ordinal, dict):
        ordinal_list = ordinal.get("result", [])
    elif isinstance(ordinal, list):
        ordinal_list = ordinal
    else:
        print(f"⚠️ WARNING: ordinal invalid type: {type(ordinal)}")
        ordinal_list = []

    ordinal_fixed = []
    for o in ordinal_list:
        if isinstance(o, dict):
            ordinal_fixed.append({
                "Architecture": o.get("architecture", "Unknown"),
                "MatchedNFRs": o.get("matched_nfrs", 0)
            })

    # ===============================
    # 2️⃣ Normalize BINARY
    # ===============================
    binary_fixed = binary.get("top_5_architectures", [])

    # ===============================
    # 3️⃣ Hybrid aggregation
    # ===============================
    result = hybrid_aggregation(
        functional=functional,
        ordinal=ordinal_fixed,
        binary=binary_fixed,
        weighted=weighted
    )

    # ===============================
    # 4️⃣ Select final architecture
    # ===============================
    selected_architecture = None

    if isinstance(result, dict):
        selected_architecture = result.get("selected_architecture")

    if not selected_architecture and isinstance(result, list) and len(result) > 0:
        selected_architecture = result[0].get("architecture")

    if not selected_architecture:
        selected_architecture = "Unknown"

    # ===============================
    # 5️⃣ Persist
    # ===============================
    save_hybrid_result(
        project_id=project_id,
        result=result,
        selected_architecture=selected_architecture
    )

    return result