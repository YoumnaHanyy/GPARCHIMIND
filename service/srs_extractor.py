import fitz  # PyMuPDF
import json
import re
import time
import logging
from typing import List, Dict


from infrastructure.repositories.extraction_repository import ExtractionRepository
import requests
import os


logger = logging.getLogger("srs_extractor")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MAX_RETRIES = 3

FALLBACK_FR_PATTERNS = [
    r"[Tt]he system shall\b[^.]*",
    r"[Tt]he system must\b[^.]*",
    r"[Tt]he user can\b[^.]*",
    r"[Uu]sers? should be able to\b[^.]*",
]


class GroqRateLimitError(Exception):
    pass


def _parse_groq_wait_time(error_message: str) -> float:
    """Return suggested retry-after seconds from a Groq rate-limit message, or None."""
    match = re.search(r"try again in ([\d.]+)s", error_message, re.IGNORECASE)
    return float(match.group(1)) + 0.5 if match else None


def generate(prompt: str) -> str:
    """Call Groq with exponential-backoff retry on rate-limit errors."""
    last_error = None
    for attempt in range(GROQ_MAX_RETRIES):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        data = response.json()
        error_obj = data.get("error", {})
        is_rate_limit = (
            response.status_code == 429
            or error_obj.get("code") == "rate_limit_exceeded"
        )
        if is_rate_limit:
            error_msg = error_obj.get("message", "")
            suggested = _parse_groq_wait_time(error_msg)
            wait_time = suggested if suggested is not None else (2 ** attempt) * 2
            last_error = GroqRateLimitError(
                f"rate_limit_exceeded (attempt {attempt + 1}/{GROQ_MAX_RETRIES}): {error_msg}"
            )
            logger.warning(
                "Groq rate limit hit (attempt %d/%d), waiting %.1fs",
                attempt + 1, GROQ_MAX_RETRIES, wait_time,
            )
            if attempt < GROQ_MAX_RETRIES - 1:
                time.sleep(wait_time)
                continue
            raise last_error
        if "choices" not in data:
            raise ValueError(f"Groq API error: {data}")
        return data["choices"][0]["message"]["content"]
    raise last_error
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
    # FALLBACK FR EXTRACTION (REGEX)
    # -------------------------------
    @staticmethod
    def _fallback_extract_functional(srs_text: str) -> List[Dict]:
        """Extract FRs via regex patterns when Groq is unavailable."""
        text = srs_text.replace("\n", " ")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: set = set()
        results = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            for pattern in FALLBACK_FR_PATTERNS:
                if re.search(pattern, sentence):
                    if sentence not in seen:
                        seen.add(sentence)
                        results.append({
                            "description": sentence,
                            "source": {"page": None, "start_index": None},
                        })
                    break
        return results

    # -------------------------------
    # FUNCTIONAL + NON-FUNCTIONAL EXTRACTION
    # -------------------------------
    def extract_requirements(self, srs_text: str, project_id: int = 2) -> Dict:
        project_name = self.extract_project_name(srs_text)
        groq_error = None
        extracted = None

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
            groq_error = e
            logger.warning(
                "Groq extraction failed (%s: %s), trying regex fallback",
                type(e).__name__, e,
            )

        if extracted is not None:
            functional_requirements = [
                {"description": f, "source": {"page": None, "start_index": None}}
                for f in extracted.get("functional", [])
            ]
            non_functional_requirements = extracted.get("non_functional", [])
            logger.info("FR source: Groq (%d items)", len(functional_requirements))
        else:
            functional_requirements = self._fallback_extract_functional(srs_text)
            non_functional_requirements = []
            logger.info("FR source: regex fallback (%d items found)", len(functional_requirements))

        if not functional_requirements and groq_error is not None:
            logger.error("❌ Groq failed and regex fallback returned no FRs")
            return {
                "project_name": project_name,
                "functional": [],
                "non_functional": [],
                "extraction_failed": True,
                "error": "FR extraction failed after retries and fallback",
                "details": str(groq_error),
            }

        ExtractionRepository.save_functional(project_id, functional_requirements)
        ExtractionRepository.save_non_functional(project_id, non_functional_requirements)
        paths = ExtractionRepository.save_extraction_results(
            project_id=project_id,
            fr=functional_requirements,
            nfr=non_functional_requirements,
        )

        return {
            "project_name": project_name,
            "functional": functional_requirements,
            "non_functional": non_functional_requirements,
            "fr_source": "groq" if extracted is not None else "fallback",
            "saved_files": paths,
        }