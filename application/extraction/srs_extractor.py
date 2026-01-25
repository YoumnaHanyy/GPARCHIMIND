import fitz  # PyMuPDF
import json
import re
import logging
from typing import List, Dict
from huggingface_hub import InferenceClient

from infrastructure.repositories.extraction_repository import ExtractionRepository


logger = logging.getLogger("srs_extractor")

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CHARS = 12000
CHUNK_SIZE = 4000


class SRSExtractor:
    def __init__(self, hf_api_key: str):
        self.client = InferenceClient(
            model=MODEL_NAME,
            token=hf_api_key,
            timeout=120
        )

    # -------------------------------
    # PDF TEXT EXTRACTION
    # -------------------------------
    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # -------------------------------
    # JSON PARSING (LLM SAFE)
    # -------------------------------
    def extract_json_from_model_output(self, output: str) -> Dict:
        cleaned = re.sub(r"```json|```", "", output).strip()

        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model output")

        stack = []
        in_string = False

        for i in range(start, len(cleaned)):
            ch = cleaned[i]

            if ch == '"' and cleaned[i - 1] != "\\":
                in_string = not in_string

            if not in_string:
                if ch == "{":
                    stack.append("{")
                elif ch == "}":
                    stack.pop()
                    if not stack:
                        json_text = cleaned[start:i + 1]
                        return json.loads(json_text)

        raise ValueError("Unbalanced JSON in model output")

    # -------------------------------
    # FUNCTIONAL + NON-FUNCTIONAL EXTRACTION
    # -------------------------------
    def extract_requirements(self, srs_text: str, project_id: int = 2) -> Dict:
        try:
            prompt = f"""
You are an expert software analyst.
Extract Functional and Non-Functional Requirements from the SRS below.

Return ONLY valid JSON:

{{
  "functional": [
    {{
      "title": "exact title",
      "description": "exact sentence",
      "source": {{ "page": null, "start_index": null }}
    }}
  ],
  "non_functional": [
    {{
      "title": "exact title",
      "description": "rewrite professionally using MUST / SHALL / SHOULD / MAY",
      "source": {{ "page": null, "start_index": null }}
    }}
  ]
}}

SRS:
{srs_text[:MAX_CHARS]}
"""

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Return JSON only. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4000
            )

            output = response.choices[0].message["content"]
            extracted = self.extract_json_from_model_output(output)

        except Exception as e:
            logger.exception("❌ HuggingFace extraction failed")
            raise RuntimeError(f"LLM extraction failed: {e}")

        # -------------------------------
        # SPLIT RESULTS
        # -------------------------------
        functional_requirements = extracted.get("functional", [])
        non_functional_requirements = extracted.get("non_functional", [])

        # -------------------------------
        # SAVE TO DATABASE
        # -------------------------------
        ExtractionRepository.save_functional(
            project_id,
            functional_requirements
        )

        ExtractionRepository.save_non_functional(
            project_id,
            non_functional_requirements
        )

        # -------------------------------
        # SAVE TO JSON FILES
        # -------------------------------
        paths = ExtractionRepository.save_extraction_results(
            project_id=project_id,
            fr=functional_requirements,
            nfr=non_functional_requirements
        )

        # -------------------------------
        # RETURN FINAL RESULT
        # -------------------------------
        return {
            "functional": functional_requirements,
            "non_functional": non_functional_requirements,
            "saved_files": paths
        }
