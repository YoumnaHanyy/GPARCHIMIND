import pandas as pd
import json
import os
from datetime import datetime
from infrastructure.database import db
from ai.utils.nfr_mapping import get_nfr_label, get_nfr_code

BASE_OUTPUT = "data/outputs"
OUTPUT_PATH = os.path.join(BASE_OUTPUT, "nfr_predictions_type_level.json")


class NFRDatasetRepository:
    """Repository for NFR training dataset"""
    
    @staticmethod
    def load_nfr_dataset_from_mongo():
        collection = db.merged_NFR_cleaned_no_dots
        docs = list(collection.find({}, {
            "_id": 0,
            "Requirement": 1,
            "Type": 1,
            "Level": 1
        }))
        return pd.DataFrame(docs)


class NFRPredictionRepository:
    """Repository for NFR predictions - keeps MongoDB and JSON in sync"""
    
    @staticmethod
    def save_batch(project_id: str, predictions: list):
        """
        Save multiple predictions to BOTH MongoDB and JSON file
        """
        if not predictions:
            return
        
        # 1️⃣ Save to MongoDB
        for p in predictions:
            db.nfr_predictions.update_one(
                {
                    "project_id": project_id,
                    "description": p["description"]
                },
                {
                    "$set": {
                        "project_id": project_id,
                        "title": p.get("title"),
                        "description": p["description"],
                        "predicted_type": p["predicted_type"],
                        "predicted_type_label": p["predicted_type_label"],
                        "confidence": p["confidence"],
                        "predicted_level": p["predicted_level"],
                        "confirmed": p.get("confirmed", False),
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        
        # 2️⃣ Sync to JSON file (all predictions for this project)
        NFRPredictionRepository._sync_to_json(project_id)
        
        print(f"✅ Saved {len(predictions)} NFR predictions to MongoDB and JSON")
    
    @staticmethod
    def get_by_project(project_id: str):
        """
        Get all NFR predictions for a project from MongoDB
        Returns list of dicts (ready to use)
        """
        docs = list(db.nfr_predictions.find(
            {"project_id": project_id},
            {"_id": 0}  # Exclude MongoDB _id
        ))
        return docs
    
    @staticmethod
    def get_high_confidence(project_id: str, threshold: float = 0.80):
        """Get only high confidence predictions"""
        docs = list(db.nfr_predictions.find(
            {
                "project_id": project_id,
                "confidence": {"$gte": threshold}
            },
            {"_id": 0}
        ))
        return docs
    
    @staticmethod
    def get_low_confidence(project_id: str, threshold: float = 0.80):
        """Get only low confidence predictions"""
        docs = list(db.nfr_predictions.find(
            {
                "project_id": project_id,
                "confidence": {"$lt": threshold}
            },
            {"_id": 0}
        ))
        return docs
    
    @staticmethod
    def confirm_nfr(
        project_id: str,
        description: str,
        confirmed_type: str,
        predicted_level: str
    ):
        """
        Update a single NFR prediction with user confirmation
        Updates BOTH MongoDB and JSON file
        Uses NFR_MAP to get full label from code
        
        Args:
            confirmed_type: NFR code (e.g., "PE", "SE", "US")
        """
        # Get full label from code using mapping
        confirmed_type_label = get_nfr_label(confirmed_type)
        
        result = db.nfr_predictions.update_one(
            {
                "project_id": project_id,
                "description": description
            },
            {
                "$set": {
                    "predicted_type": confirmed_type,
                    "predicted_type_label": confirmed_type_label,  # Use mapped label
                    "predicted_level": predicted_level,
                    "confirmed": True,
                    "confidence": 1.0,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            # Sync to JSON file after update
            NFRPredictionRepository._sync_to_json(project_id)
            print(f"✅ Confirmed NFR ({confirmed_type} -> {confirmed_type_label}): {description[:50]}...")
        else:
            print(f"⚠️ NFR not found for confirmation: {description[:50]}...")
        
        return result.modified_count > 0
    
    @staticmethod
    def _sync_to_json(project_id: str):
        """
        Internal method: Sync MongoDB data to JSON file
        Writes all predictions for the given project to the JSON file
        """
        try:
            # Get all predictions for this project from MongoDB
            all_predictions = NFRPredictionRepository.get_by_project(project_id)
            
            # Remove MongoDB-specific fields and datetime objects
            clean_predictions = []
            for pred in all_predictions:
                clean_pred = dict(pred)
                clean_pred.pop("_id", None)
                clean_pred.pop("project_id", None)
                clean_pred.pop("updated_at", None)
                clean_predictions.append(clean_pred)
            
            # Write to JSON file
            os.makedirs(BASE_OUTPUT, exist_ok=True)
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(clean_predictions, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Synced {len(clean_predictions)} predictions to JSON file")
            
        except Exception as e:
            print(f"⚠️ Failed to sync to JSON file: {e}")