import fitz  # PyMuPDF
import json
import re
import logging
from typing import List, Dict


from infrastructure.repositories.extraction_repository import ExtractionRepository
import requests
import os


logger = logging.getLogger("srs_extractor")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate(prompt: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        },
        timeout=60
    )

    data = response.json()

    if "choices" not in data:
        raise ValueError(f"Groq API error: {data}")

    return data["choices"][0]["message"]["content"]
MAX_CHARS = 12000
CHUNK_SIZE = 4000


class SRSExtractor:
    KEYWORDS = ["shall", "must", "will", "should", "can", "ought to","needs to" , "need to", "might"]
    
    def __init__(self, hf_api_key: str):
        pass

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
    def extract_candidates(self, text: str):
      text = text.replace("\n", " ")
      sentences = re.split(r'\.', text)
      candidates = []

      for s in sentences:
        if any(k in s.lower() for k in self.KEYWORDS):
            clean = s.strip()
            if len(clean) > 10:
                candidates.append(clean)

      return list(set(candidates))
    def extract_project_name(self, text: str) -> str:
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for i, line in enumerate(lines):
            lower_line = line.lower()

            if "software requirement specification document for" in lower_line:
            
            # ✅ خد الجزء اللي بعد "for" من نفس السطر
                after_for = line.split("for", 1)[-1].strip()

                title_parts = []

                if after_for:
                    title_parts.append(after_for)

            # ✅ كمل السطور اللي بعده
                for j in range(i + 1, min(i + 6, len(lines))):
                    current = lines[j]

                    if (
                        "," in current
                        or "supervised" in current.lower()
                        or "table" in current.lower()
                        or any(char.isdigit() for char in current)
                    ):
                        break
 
                    title_parts.append(current)
  
                if title_parts:
                    return " ".join(title_parts)

        return "Unknown Project" 
    # -------------------------------
    # JSON PARSING (LLM SAFE)
    # -------------------------------
    def extract_json_from_model_output(self, output: str) -> Dict:
        """
        Robust JSON extraction from LLM output.
        Handles malformed outputs gracefully.
        """

        # remove markdown fences
        cleaned = re.sub(r"```json|```", "", output).strip()

        # 1️⃣ Try direct JSON parsing
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 2️⃣ Fallback: extract first valid JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                pass

        # 3️⃣ Fail safely
        raise ValueError("LLM returned invalid or unbalanced JSON")

    # -------------------------------
    # FUNCTIONAL + NON-FUNCTIONAL EXTRACTION
    # -------------------------------
    def extract_requirements(self, srs_text: str, project_id: int = 2) -> Dict:
        project_name = self.extract_project_name(srs_text)
        try:
            candidates = self.extract_candidates(srs_text)

            joined_text = "\n".join(candidates[:200])

            prompt = f"""
You are a software requirements analyst.

Classify the following into:

1. Functional Requirements → RETURN AS LIST OF STRINGS ONLY
2. Non-Functional Requirements → RETURN OBJECTS WITH title + description

Return ONLY JSON:

{{
  "functional": ["sentence1", "sentence2"],
  "non_functional": [
    {{
      "title": "",
      "description": ""
    }}
  ]
}}

Requirements:
{joined_text}
"""

            output = generate(prompt)
            extracted = self.extract_json_from_model_output(output)

        except Exception as e:
            logger.exception("❌ Groq extraction failed")

            # 🔹 Graceful failure (system does NOT crash)
            return {
                "project_name": project_name,
                "functional": [],
                "non_functional": [],
                "error": "LLM extraction failed",
                "details": str(e)
            }

        # -------------------------------
        # SPLIT RESULTS
        # -------------------------------
        functional_requirements = [
    {
        "description": f,
        "source": {"page": None, "start_index": None}
    }
    for f in extracted.get("functional", [])
]
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
            "project_name": project_name,
            "functional": functional_requirements,
            "non_functional": non_functional_requirements,
            "saved_files": paths
        }