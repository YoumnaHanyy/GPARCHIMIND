from infrastructure.database import db

architecture_collection = db["ArchitectureDataset"]

def get_architecture_dataset():
    """
    Load architecture dataset from MongoDB
    """
    cursor = architecture_collection.find(
        {},
        {
            "_id": 0,
            "Architecture": 1,
            "Type": 1,
            "Level": 1,
            "LevelNorm": 1   # ✅ مهم للـ Weighted
        }
    )

    data = list(cursor)

    if not data:
        raise ValueError("ArchitectureDataset collection is empty")

    return data
