from infrastructure.database import db
import copy
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BASE_OUTPUT = os.path.join(BASE_DIR, "data", "outputs")


class ExtractionRepository:

    @staticmethod
    def save_functional(project_id: int, functional_reqs: list):
        db.fr_extracted.delete_many({"project_id": project_id})

        if not functional_reqs:
            return

        for fr in functional_reqs:
            fr["project_id"] = project_id

        db.fr_extracted.insert_many(copy.deepcopy(functional_reqs))

    @staticmethod
    def save_non_functional(project_id: int, non_functional_reqs: list):
        db.nfr_extracted.delete_many({"project_id": project_id})

        if not non_functional_reqs:
            return

        for nfr in non_functional_reqs:
            nfr["project_id"] = project_id

        db.nfr_extracted.insert_many(copy.deepcopy(non_functional_reqs))

    # ✅ SAVE DIRECTLY IN data/outputs
    @staticmethod
    def save_extraction_results(project_id: int, fr: list, nfr: list):
        os.makedirs(BASE_OUTPUT, exist_ok=True)

        fr_path = os.path.join(BASE_OUTPUT, "functional_requirements.json")
        nfr_path = os.path.join(BASE_OUTPUT, "non_functional_requirements.json")

        with open(fr_path, "w", encoding="utf-8") as f:
            json.dump(fr, f, indent=2, ensure_ascii=False)

        with open(nfr_path, "w", encoding="utf-8") as f:
            json.dump(nfr, f, indent=2, ensure_ascii=False)

        print("✅ Saved directly to data/outputs")

        return {
            "functional": fr_path,
            "non_functional": nfr_path
        }
    
    @staticmethod
    def get_functional(project_id: int):
        """
        Read functional requirements from MongoDB
        (NO extraction, NO LLM)
        """
        return list(
            db.fr_extracted.find(
                {"project_id": project_id},
                {"_id": 0}
            )
        )

