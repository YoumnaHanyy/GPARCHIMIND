import fitz  # PyMuPDF
import json
import re
import logging
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient

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
            raise ValueError("No JSON object found")

        stack = []
        in_string = False

        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if c == '"' and cleaned[i - 1] != "\\":
                in_string = not in_string
            if not in_string:
                if c == "{":
                    stack.append("{")
                elif c == "}":
                    stack.pop()
                    if not stack:
                        json_text = cleaned[start:i + 1]
                        return json.loads(json_text)

        raise ValueError("Unbalanced JSON")

    # -------------------------------
    # FUNCTIONAL + NON-FUNCTIONAL EXTRACTION
    # -------------------------------
    def extract_requirements(self, srs_text: str) -> Dict:
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
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message["content"]
        return self.extract_json_from_model_output(output)
