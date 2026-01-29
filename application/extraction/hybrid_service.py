from ai.methods.hybrid_method import hybrid_aggregation



def execute_hybrid_method(project_id, functional, ordinal, binary, weighted):

    # =====================================================
    # 1️⃣ Normalize ORDINAL
    # =====================================================
    # ordinal ممكن ييجي dict أو list
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

    # =====================================================
    # 2️⃣ Normalize BINARY
    # =====================================================
    # binary دايمًا dict عندك
    binary_fixed = binary.get("top_5_architectures", [])

    # =====================================================
    # 3️⃣ Call Hybrid Aggregation
    # =====================================================
    result = hybrid_aggregation(
        functional=functional,
        ordinal=ordinal_fixed,
        binary=binary_fixed,
        weighted=weighted
    )

    
    return result
