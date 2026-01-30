from datetime import datetime
from infrastructure.database import db
from ai.utils.nfr_mapping import get_nfr_label, get_nfr_code

CONFIRMED_NFR_COLLECTION = db.confirmed_NFR


def load_confirmed_types():
    """
    Returns:
    {
      description: confirmed_type_code
    }
    Example: {"The system must...": "PE"}
    """
    docs = CONFIRMED_NFR_COLLECTION.find({})
    return {
        doc["description"]: doc["confirmed_type"]
        for doc in docs
    }


def save_confirmed_types(mapping: dict):
    """
    mapping:
    {
      description: confirmed_type_code
    }
    Example: {"The system must...": "PE"}
    
    Stores both code and label in MongoDB
    """
    for desc, code in mapping.items():
        label = get_nfr_label(code)
        
        CONFIRMED_NFR_COLLECTION.update_one(
            {"description": desc},
            {
                "$set": {
                    "confirmed_type": code,
                    "confirmed_type_label": label,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )


def save_user_feedback(rows: list):
    """
    rows:
    [
      {
        title,
        description,
        type  # This is the NFR code (e.g., "PE", "SE")
      }
    ]
    
    Stores both code and label in MongoDB
    """
    if not rows:
        return

    docs = []
    ts = datetime.utcnow()

    for r in rows:
        code = r.get("type", "")
        label = get_nfr_label(code)
        
        docs.append({
            "title": r.get("title", ""),
            "description": r.get("description"),
            "confirmed_type": code,
            "confirmed_type_label": label,
            "timestamp": ts
        })

    CONFIRMED_NFR_COLLECTION.insert_many(docs)


def save_confirmed_nfr(doc: dict):
    """
    Save a single confirmed NFR
    
    Args:
        doc: {
            "project_id": str,
            "description": str,
            "confirmed_type": str,  # NFR code (e.g., "PE")
            "predicted_level": str
        }
    """
    code = doc["confirmed_type"]
    label = get_nfr_label(code)
    
    db.confirmed_NFR.update_one(
        {
            "project_id": doc["project_id"],
            "description": doc["description"]
        },
        {
            "$set": {
                "confirmed_type": code,
                "confirmed_type_label": label,
                "predicted_level": doc["predicted_level"],
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )