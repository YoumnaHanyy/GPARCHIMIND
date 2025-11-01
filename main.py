from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import json
import os
import re

MAX_CHARS = 12000 
# ----------------------------

app = FastAPI()

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# huggingface client
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_srs/")
async def upload_srs(file: UploadFile = File(...)):
    filename = file.filename
    safe_path = os.path.join("uploads", filename)

    # save uploaded PDF
    with open(safe_path, "wb") as f:
        f.write(await file.read())

    # extract raw text
    text = extract_text_from_pdf(safe_path)
    text = text[:MAX_CHARS]


 # after extracting and trimming text
    # after extracting and trimming text
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
2. Non-functional requirements → 
   - Rephrase the description professionally, 
   - Use one of these modal verbs to indicate importance:
     - "must" → critical requirement
     - "shall" → mandatory standard requirement
     - "should" → recommended requirement
     - "may" → optional feature
     - "can" → possible capability
   - Keep the title identical to what appears in the SRS (no renaming or paraphrasing).
   - Output exactly the same number of non-functional requirements as identified in the SRS (no adding or guessing new ones).
3. Do not invent or infer any requirement that is not explicitly in the text.
4. Each description must be grammatically correct, concise, and faithful to the original meaning.
5. If page or index are unknown, set them to null.
6. The output must be valid JSON only — no explanations or commentary outside the JSON.

SRS Text:
{text}
"""

    

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1500,
            temperature=0.1,
            response_format={"type": "json_object"},  # ✅ Force pure JSON
            messages=[
               
                {"role": "user", "content": prompt}
            ]
        )
        output_text = response.choices[0].message["content"]

    except Exception as e:
        return {"error": "Model request failed", "exception": str(e)}

    # ✅ Parse JSON safely
    try:
        json_match = re.search(r'\{[\s\S]*\}', output_text)
        if not json_match:
            raise ValueError("No valid JSON detected")
        json_text = json_match.group(0)
        parsed = json.loads(json_text)

        # Save results
        with open("requirements_detailed.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        # optional: save each category separately
        with open("functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("functional", []), f, indent=2, ensure_ascii=False)
        with open("non_functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("non_functional", []), f, indent=2, ensure_ascii=False)

        return parsed
    except Exception as e:
        # save raw for debugging
        with open("requirements_raw.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        return {"error": "Failed to parse JSON from model output", "raw_output": output_text, "exception": str(e)}