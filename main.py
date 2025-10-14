from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import json
import os

# ---------- CONFIG ----------
# replace with your token
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# ----------------------------

app = FastAPI()

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# huggingface client
client = InferenceClient(api_key=HF_API_KEY)

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
    # save file
    filename = file.filename
    safe_path = os.path.join("uploads", filename)
    with open(safe_path, "wb") as f:
        f.write(await file.read())

    # extract text
    text = extract_text_from_pdf(safe_path)

    # build prompt for detailed requirements (title + description)
    prompt = f"""
You are an expert software analyst.
Extract both Functional and Non-Functional Requirements from the following Software Requirements Specification (SRS) text.

For each requirement produce:
- title (short)
- description (detailed sentence or two)

Return a valid JSON object like:
{{
  "functional": [
    {{"title":"User Authentication","description":"The system shall..."}},
    ...
  ],
  "non_functional": [
    {{"title":"Performance","description":"The system shall..."}},
    ...
  ]
}}

SRS Text:
{text}
"""

    # Use chat completions (the model expects conversational)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert software analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )

    # get assistant text
    try:
        output_text = response.choices[0].message["content"]
    except Exception:
        # fallback if provider returns different format
        output_text = str(response)

    # try parse JSON
    try:
        parsed = json.loads(output_text)
        # save separate files
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
