import numpy as np
import pandas as pd

# MongoDB
from infrastructure.database import db
from ai.methods.strategy import ArchitectureAnalysisStrategy



# ============================================================
# 1️⃣ Configuration
# ============================================================

NFR_ORDER = ["PE", "SC", "MN", "A", "SE", "US", "PO", "O"]

ARCH_COLLECTION = "ArchitectureDataset"
BINARY_VECTOR_COLLECTION = "binary_results"  # optional if saved


# ============================================================
# 2️⃣ Load Architecture Dataset from MongoDB
# ============================================================

def load_architecture_dataset():
    collection = db[ARCH_COLLECTION]

    docs = list(
        collection.find(
            {},
            {"_id": 0, "Architecture": 1, "Type": 1, "label": 1}
        )
    )

    # 🧠 build binary vectors manually
    arch_map = {}

    for doc in docs:
        arch = doc.get("Architecture")
        nfr = doc.get("Type")
        label = doc.get("label", 0)

        if not arch or nfr not in NFR_ORDER:
            continue

        if arch not in arch_map:
            arch_map[arch] = {k: 0 for k in NFR_ORDER}

        arch_map[arch][nfr] = int(label)

    # convert to DataFrame
    rows = []
    for arch, vec in arch_map.items():
        row = {"Architecture": arch}
        row.update(vec)
        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# 3️⃣ Compute Architecture Scores (Binary Distance)
# ============================================================

def compute_architecture_scores(binary_vector, arch_df):
    """
    Compare SRS binary vector with each architecture vector
    Using Hamming distance (normalized)
    """
    results = []

    for _, row in arch_df.iterrows():
        arch_name = row["Architecture"]

        arch_vec = row[NFR_ORDER].values.astype(int)
        srs_vec = np.array([binary_vector[nfr] for nfr in NFR_ORDER])

        diff = np.sum(np.abs(arch_vec - srs_vec))
        score = 1 - (diff / len(NFR_ORDER))

        results.append({
            "architecture": arch_name,
            "score": round(float(score), 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# 4️⃣ Binary Method Main Function
# ============================================================

def run_binary_method(binary_vector, top_k=5):
    """
    binary_vector example:
    {
        "PE": 1,
        "SC": 0,
        "MN": 1,
        ...
    }
    """
    arch_df = load_architecture_dataset()
    scores = compute_architecture_scores(binary_vector, arch_df)

    return {
        "binary_vector": binary_vector,
        "top_5_architectures": scores[:top_k],
        "best_architecture": scores[0] if scores else None
    }


# ============================================================
# 5️⃣ Optional: Load Binary Vector from MongoDB
# ============================================================

def load_binary_vector_from_db():
    """
    Optional helper if binary vector is saved in MongoDB
    """
    collection = db[BINARY_VECTOR_COLLECTION]
    doc = collection.find_one({}, {"_id": 0})
    return doc


# ============================================================
# 6️⃣ Local Testing
# ============================================================

if __name__ == "__main__":

    # Example test vector (dummy)
    test_binary_vector = {
        "PE": 1,
        "SC": 1,
        "MN": 0,
        "A": 1,
        "SE": 0,
        "US": 1,
        "PO": 0,
        "O": 0
    }

    result = run_binary_method(test_binary_vector)

    print("\n=== Binary Vector ===")
    print(result["binary_vector"])

    print("\n=== Top Architectures ===")
    for arch in result["top_5_architectures"]:
        print(arch)

    print("\n=== Best Architecture ===")
    print(result["best_architecture"])


 # ============================================================
# 7️⃣ Strategy Wrapper
# ============================================================

class BinaryMethod(ArchitectureAnalysisStrategy):
    def run(self, binary_vector, top_k=5):
        return run_binary_method(binary_vector, top_k)
