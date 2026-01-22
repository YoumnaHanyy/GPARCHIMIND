# ===========================
# app.py (FULL READY COPY/PASTE)
# ===========================

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import json
import os
import re
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

MAX_CHARS = 12000
HF_API_KEY = "REDACTED_HF_TOKEN"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

USER_FEEDBACK_CSV = "user_labeled_nfr_new.csv"

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_API_KEY,
    timeout=120
)

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


# ======================
# NEW: user label endpoint
# ======================

class NFRLabelRequest(BaseModel):
    title: str | None = None
    description: str
    confirmed_type: str
    confirmed_level: str | None = None

def append_user_feedback(title: str, description: str, type_code: str, level: str):
    row = {
        "Timestamp": datetime.utcnow().isoformat(),
        "Title": title or "",
        "Requirement": description,
        "Type": type_code,
        "Level": level
    }
    df_row = pd.DataFrame([row])

    if not os.path.exists(USER_FEEDBACK_CSV):
        df_row.to_csv(USER_FEEDBACK_CSV, index=False, encoding="utf-8")
    else:
        df_row.to_csv(USER_FEEDBACK_CSV, mode="a", header=False, index=False, encoding="utf-8")

@app.post("/submit_nfr_label/")
async def submit_nfr_label(payload: NFRLabelRequest):
    append_user_feedback(
        title=payload.title or "",
        description=payload.description.strip(),
        type_code=payload.confirmed_type.strip(),
        level=(payload.confirmed_level or "Unknown").strip()
    )
    return {"status": "ok", "saved_to": USER_FEEDBACK_CSV}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_srs/")
async def upload_srs(file: UploadFile = File(...)):
    filename = file.filename
    safe_path = os.path.join("uploads", filename)

    with open(safe_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(safe_path)
    text = text[:MAX_CHARS]

    prompt = f"""
You are an expert software analyst.

Your task is to extract both Functional and Non-Functional Requirements from the following SRS text.

Return ONLY a single clean JSON object with this exact structure:

{{
  "functional": [
    {{
      "title": "<exact title as it appears in the SRS (do not modify or paraphrase)>",
      "description": "<exact sentence(s) copied verbatim from the SRS (no changes)>",
      "source": {{ "page": <page_number_if_known_or_null>, "start_index": <character_index_or_null> }}
    }}
  ],
  "non_functional": [
    {{
      "title": "<exact title as it appears in the SRS (do not create new or paraphrased titles)>",
      "description": "<professionally reworded version using 'shall', 'must', 'should', 'may', or 'can' depending on importance>",
      "source": {{ "page": <page_number_if_known_or_null>, "start_index": <character_index_or_null> }}
    }}
  ]
}}

RULES:
1. Functional requirements → copy both title and description verbatim from the SRS (no changes at all).
2. Non-functional requirements → rephrase professionally with modal verb based on importance.
3. Do not invent requirements.
4. Output valid JSON only.

SRS Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1500,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        output_text = response.choices[0].message["content"]

    except Exception as e:
        return {"error": "Model request failed", "exception": str(e)}

    try:
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        if not json_match:
            raise ValueError("No valid JSON detected")
        json_text = json_match.group(0)
        parsed = json.loads(json_text)

        with open("requirements_detailed.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        with open("functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("functional", []), f, indent=2, ensure_ascii=False)
        with open("non_functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("non_functional", []), f, indent=2, ensure_ascii=False)

        return parsed

    except Exception as e:
        with open("requirements_raw.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        return {"error": "Failed to parse JSON from model output", "raw_output": output_text, "exception": str(e)}
