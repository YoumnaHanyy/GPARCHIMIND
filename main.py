from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import json
import os
import re
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import csv
import threading

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archimind")

HF_API_KEY = "REDACTED_HF_TOKEN"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CHARS = 12000
CHUNK_SIZE = 4000

# ====== NEW: thresholds & dataset path ======
TYPE_CONFIDENCE_THRESHOLD = 0.30
DATASET_APPEND_PATH = "merged_NFR_cleaned_no_dots.csv"  # Type,Requirement,Level
DATASET_LOCK = threading.Lock()

# Allowed NFR type codes (your abbreviations)
ALLOWED_TYPES = ["A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US", "OT"]

app = FastAPI(title="ArchiMind SRS Processor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Hugging Face client (LLM) ---
client = InferenceClient(model=MODEL_NAME, token=HF_API_KEY, timeout=120)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks for processing."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks


def parse_model_json(output_text: str) -> List[Dict]:
    """Parse JSON array from model output."""
    if not output_text:
        return []

    cleaned = re.sub(r"```json\s*", "", output_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    try:
        result = json.loads(cleaned.strip())
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ["requirements", "functional", "items", "data"]:
                if key in result and isinstance(result[key], list):
                    return result[key]
        return []
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON array from model output: {output_text[:200]}...")
    return []


def extract_json_from_model_output(output: str) -> str:
    """
    Robustly extract the first JSON object from model output.
    """
    if output is None:
        raise ValueError("Model returned empty output")

    cleaned = output.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = re.sub(r"(?i)^.*?(?=\{)", "", cleaned, count=1).strip()

    start = cleaned.find("{")
    if start == -1:
        json.loads(cleaned)
        return cleaned

    brace_stack = []
    in_string = False
    escape_next = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_stack.append("{")
            elif ch == "}":
                if not brace_stack:
                    continue
                brace_stack.pop()
                if not brace_stack:
                    json_text = cleaned[start:i + 1]
                    json.loads(json_text)
                    return json_text

    raise ValueError("No balanced JSON object found in model output")


def choose_architecture(functional_reqs: List[Dict]) -> Dict:
    """Choose architecture based on functional requirements analysis."""
    text = " ".join([f"{r.get('title', '')} {r.get('description', '')}" for r in functional_reqs]).lower()
    scores = {"microservices": 0, "event_driven": 0, "soa": 0, "layered_monolith": 0, "modular": 0}

    if any(k in text for k in ["real-time", "low latency", "latency", "stream"]):
        scores["event_driven"] += 3
        scores["microservices"] += 2
    if any(k in text for k in ["high throughput", "large data", "big data", "analytics"]):
        scores["event_driven"] += 2
        scores["microservices"] += 3
        scores["modular"] += 1
    if any(k in text for k in ["integrat", "api", "third-party", "external system"]):
        scores["soa"] += 3
        scores["microservices"] += 2
    if any(k in text for k in ["independent features", "plug-in", "extensible"]):
        scores["microservices"] += 3
        scores["modular"] += 3
    if any(k in text for k in ["transactional", "strong consistency", "ACID"]):
        scores["layered_monolith"] += 2
        scores["microservices"] += 1
    if len(functional_reqs) <= 6 and not any(k in text for k in ["high", "many", "integrat", "real-time"]):
        scores["layered_monolith"] += 3

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_scores[0][1]
    chosen = [k for k, v in scores.items() if v == top_score and v > 0]

    architecture_map = {
        "microservices": {"name": "Microservices Architecture", "rationale": "Recommended for modular, scalable systems."},
        "event_driven": {"name": "Event-driven / Streaming Architecture", "rationale": "Recommended for real-time or high-throughput systems."},
        "soa": {"name": "Service-Oriented Architecture (SOA)", "rationale": "Recommended for multiple integrations and service contracts."},
        "layered_monolith": {"name": "Layered / Monolithic Architecture", "rationale": "Recommended for small to medium systems with limited features."},
        "modular": {"name": "Modular / Component-based Architecture", "rationale": "Recommended for maintainable and reusable components."}
    }

    chosen_architectures = [architecture_map[c] for c in chosen] if chosen else [architecture_map["layered_monolith"]]
    explanation = {
        "chosen_architectures": [{"key": c, "name": architecture_map[c]["name"], "rationale": architecture_map[c]["rationale"]} for c in chosen]
        if chosen else [{"key": "layered_monolith", "name": architecture_map["layered_monolith"]["name"], "rationale": architecture_map["layered_monolith"]["rationale"]}],
        "scoring": scores,
        "supporting_papers": [
            "Taylor et al., Software Architecture: Foundations (2009)",
            "Garlan & Shaw, Comparison Framework for Architecture Styles (1993)",
            "Alvaro et al., Scalability! But at what COST? (2017)",
            "Bass et al., Software Architecture in Practice",
            "Lago et al., Sustainable Software Architectures (2015)",
            "Penzenstadler et al., Designing Software for Sustainability (2013)"
        ]
    }
    return {"chosen_architectures": chosen_architectures, "explanation": explanation}


# ============================================================
# NEW: Type prediction with confidence (BERT)
# ============================================================

def load_type_model_and_encoder():
    tokenizer = BertTokenizer.from_pretrained("./trained_nfr_type_model")
    model = BertForSequenceClassification.from_pretrained("./trained_nfr_type_model")
    model.eval()

    df = pd.read_csv(DATASET_APPEND_PATH)
    le_type = LabelEncoder()
    le_type.fit(df["Type"].astype(str))

    return tokenizer, model, le_type


def predict_type_with_topk(tokenizer, model, le_type, text: str, top_k: int = 3) -> Dict[str, Any]:
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**tokens)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()

    top_idx = np.argsort(-probs)[:top_k]
    top = []
    for idx in top_idx:
        label = le_type.inverse_transform([int(idx)])[0]
        top.append({"type": str(label), "confidence": float(probs[idx])})

    best = top[0]
    return {
        "best_type": best["type"],
        "best_confidence": best["confidence"],
        "top": top
    }


# ============================================================
# NEW: Level prediction using LLM ONLY
# ============================================================

def predict_level_with_llm(requirement_text: str, chosen_type: str) -> str:
    """
    Returns one of: High / Medium / Low
    """
    prompt = f"""
You are an expert software requirements analyst.

Task:
Given a non-functional requirement (NFR) and its type code, classify its LEVEL as exactly one of:
- "High"  (very strict / demanding / critical)
- "Medium" (moderate)
- "Low"   (weak / flexible / minimal)

Rules:
- Output ONLY valid JSON: {{"level":"High"}} (or Medium/Low)
- Consider requirement strictness based on wording (MUST/SHALL/SHOULD), measurable targets, risk, and criticality.
- Do not output explanations.

NFR Type Code: {chosen_type}
NFR Text: {requirement_text}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=80,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message["content"]
    js = json.loads(raw)
    level = str(js.get("level", "Medium")).strip()

    if level not in ["High", "Medium", "Low"]:
        level = "Medium"
    return level


# ============================================================
# NEW: Append confirmed NFR to dataset
# ============================================================

def append_to_dataset(type_code: str, requirement_text: str, level: str):
    """
    Appends a new row to merged_NFR_cleaned_no_dots.csv with columns: Type,Requirement,Level
    """
    if type_code not in ALLOWED_TYPES:
        raise ValueError(f"Invalid type code: {type_code}")

    if level not in ["High", "Medium", "Low"]:
        raise ValueError(f"Invalid level: {level}")

    # Ensure file exists with header
    with DATASET_LOCK:
        file_exists = os.path.exists(DATASET_APPEND_PATH)
        if not file_exists:
            with open(DATASET_APPEND_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Type", "Requirement", "Level"])

        # Append row
        with open(DATASET_APPEND_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([type_code, requirement_text, level])


# ============================================================
# Ordinal Method (UPDATED): Type from BERT, Level from LLM
# ============================================================

def run_ordinal_method(nfr_json_path="non_functional_requirements.json"):
    try:
        tokenizer, model_type, le_type = load_type_model_and_encoder()

        with open(nfr_json_path, "r", encoding="utf-8") as f:
            nfrs = json.load(f)

        results = []
        for item in nfrs:
            desc = item.get("description", "")
            type_pred = predict_type_with_topk(tokenizer, model_type, le_type, desc, top_k=3)
            chosen_type = type_pred["best_type"]

            # ✅ Level via LLM ONLY
            level_pred = predict_level_with_llm(desc, chosen_type)

            results.append({
                "title": item.get("title"),
                "description": desc,
                "predicted_type": chosen_type,
                "predicted_level": level_pred,
                "type_confidence": type_pred["best_confidence"],
                "type_top": type_pred["top"]
            })

        with open("nfr_predictions_type_level.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Recommend architectures
        pred_df = pd.DataFrame(results).rename(columns={"predicted_type": "Type", "predicted_level": "Level"})
        arch_df = pd.read_csv("ArchitectureDataset.csv")

        matches = pred_df.merge(arch_df, on=["Type", "Level"], how="inner")
        style_scores = matches["Architecture"].value_counts()

        top_arch = [{"Architecture": style, "MatchedNFRs": int(score)} for style, score in style_scores.head(5).items()]

        with open("Ordinal_Method_Top_Arch.json", "w", encoding="utf-8") as f:
            json.dump(top_arch, f, indent=2, ensure_ascii=False)

        return top_arch
    except Exception as e:
        logger.exception(f"Ordinal method failed: {e}")
        return []


# ============================================================
# (Binary + Weighted + Hybrid) — keep as you had
# ============================================================

def run_binary_method(nfr_json_path="non_functional_requirements.json"):
    try:
        tokenizer = BertTokenizer.from_pretrained("./trained_nfr_binary_model")
        model = BertForSequenceClassification.from_pretrained("./trained_nfr_binary_model")
        model.eval()

        arch_df = pd.read_csv("architecture_datasetBinary (1).csv")
        NFR_ORDER = ["PE", "SC", "MN", "A", "SE", "US", "PO", "O"]

        def predict_nfr(text):
            tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                output = model(**tokens)
            return torch.argmax(output.logits).item()

        with open(nfr_json_path, "r", encoding="utf-8") as f:
            extracted_nfrs = json.load(f)

        srs = [item["description"] for item in extracted_nfrs]

        srs_vector = {k: 0 for k in NFR_ORDER}
        for sentence in srs:
            for nfr_cat in NFR_ORDER:
                if nfr_cat.lower() in sentence.lower():
                    srs_vector[nfr_cat] = predict_nfr(sentence)

        results = []
        for _, row in arch_df.iterrows():
            arch = row["Architecture Style"]
            arch_vec = row[NFR_ORDER].values.astype(int)
            srs_vec = np.array([srs_vector[k] for k in NFR_ORDER], dtype=int)
            diff = np.sum(np.abs(arch_vec - srs_vec))
            score = 1 - (diff / len(NFR_ORDER))
            results.append((arch, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

        output = {"srs_vector": srs_vector, "top_5_architectures": results[:5], "best_architecture": results[0]}
        with open("Binary_method_Top_arch.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        return results[:5]
    except Exception as e:
        logger.exception(f"Binary method failed: {e}")
        return []


def run_weighted_score_method(nfr_json_path="non_functional_requirements.json"):
    try:
        MODEL_DIR = "./trained_nfr_model"
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()

        df = pd.read_csv("merged_NFR_cleaned.csv")
        le = LabelEncoder()
        le.fit(df["Type"])
        NFR_CATEGORIES = list(le.classes_)

        def predict_nfr(sentence):
            tokens = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokens)
                pred_idx = torch.argmax(outputs.logits, dim=1).item()
            return le.inverse_transform([pred_idx])[0]

        def requirement_strength(description):
            s = description.lower()
            if "must" in s:
                return 3.0
            if "shall" in s:
                return 2.0
            if "should" in s:
                return 1.0
            return 0.5

        with open(nfr_json_path, "r", encoding="utf-8") as f:
            extracted_nfrs = json.load(f)

        freq_counts = {cat: 0 for cat in NFR_CATEGORIES}
        must_scores = {cat: 0.0 for cat in NFR_CATEGORIES}
        predicted_nfrs = []

        for item in extracted_nfrs:
            desc = item["description"]
            nfr_type = predict_nfr(desc)
            predicted_nfrs.append(nfr_type)
            strength = requirement_strength(desc)
            freq_counts[nfr_type] += 1
            must_scores[nfr_type] += strength

        existing_nfrs = set(predicted_nfrs)
        freq_counts = {k: v for k, v in freq_counts.items() if k in existing_nfrs}
        must_scores = {k: v for k, v in must_scores.items() if k in existing_nfrs}

        max_freq = max(freq_counts.values()) or 1
        freq_norm = {k: v / max_freq for k, v in freq_counts.items()}

        max_must = max(must_scores.values()) or 1
        must_norm = {k: v / max_must for k, v in must_scores.items()}

        # importance from SRS (keep as-is)
        import PyPDF2
        import nltk
        nltk.download("punkt", quiet=True)

        pdf_path = "uploads/SRS.pdf"
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"

        sentences = nltk.sent_tokenize(text)
        counts = {t: 0 for t in NFR_CATEGORIES}
        for s in sentences:
            nfr = predict_nfr(s)
            counts[nfr] += 1

        total_sentences = len(sentences) or 1
        importance = {k: round(v / total_sentences, 4) for k, v in counts.items()}

        total_weight = {}
        for nfr in freq_norm.keys():
            total_weight[nfr] = (
                0.333 * freq_norm.get(nfr, 0) +
                0.333 * must_norm.get(nfr, 0) +
                0.333 * importance.get(nfr, 0)
            )

        ssum = sum(total_weight.values()) or 1
        total_weight = {k: round(v / ssum, 4) for k, v in total_weight.items()}

        df_arch = pd.read_csv("ArchitectureDataset2.csv")
        arch_scores = {}

        for _, row in df_arch.iterrows():
            arch = row["Architecture"]
            nfr = row["Type"]
            if nfr not in total_weight:
                continue
            level = row.get("LevelNorm", row["LevelNorm"])
            weight = total_weight[nfr]
            arch_scores[arch] = arch_scores.get(arch, 0) + level * weight

        top_arch = sorted(
            [{"Architecture": k, "Score": round(v, 4)} for k, v in arch_scores.items()],
            key=lambda x: x["Score"],
            reverse=True
        )[:5]

        results = {
            "normalized_frequency": freq_norm,
            "normalized_must_score": must_norm,
            "importance_score": importance,
            "total_nfr_weights": total_weight,
            "top_architectures": top_arch
        }

        with open("Weighted_Score_Top_Arch.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
    except Exception as e:
        logger.exception(f"Weighted score method failed: {e}")
        return {}


def normalize_scores(score_dict):
    if not score_dict:
        return {}
    max_val = max(score_dict.values()) or 1
    return {k: v / max_val for k, v in score_dict.items()}


def hybrid_aggregation(functional, ordinal, binary, weighted):
    final_scores = {}
    architectures = set()

    for item in ordinal:
        architectures.add(item["Architecture"])

    for arch, _ in binary:
        architectures.add(arch)

    for item in weighted.get("top_architectures", []):
        architectures.add(item["Architecture"])

    weighted_scores = [a["Score"] for a in weighted.get("top_architectures", [])]
    max_weighted = max(weighted_scores) if weighted_scores else 1
    functional_scores = functional.get("explanation", {}).get("scoring", {})
    max_f = max(functional_scores.values()) if functional_scores else 1

    for arch in architectures:
        raw_f = functional_scores.get(arch.lower(), 0)
        s_f = raw_f / max_f

        s_o = next((a["MatchedNFRs"] for a in ordinal if a["Architecture"] == arch), 0)
        s_b = next((score for name, score in binary if name == arch), 0)

        raw_w = next((a["Score"] for a in weighted.get("top_architectures", []) if a["Architecture"] == arch), 0)
        s_w = raw_w / max_weighted

        final_scores[arch] = (0.20 * s_f + 0.25 * s_o + 0.20 * s_b + 0.35 * s_w)

    final_scores = normalize_scores(final_scores)

    return sorted(
        [{"Architecture": k, "FinalScore": round(v * 100, 2)} for k, v in final_scores.items()],
        key=lambda x: x["FinalScore"],
        reverse=True
    )[:5]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# NEW endpoint: confirm & save low confidence Type
# ============================================================

@app.post("/confirm_nfr/")
async def confirm_nfr(payload: Dict[str, Any]):
    """
    payload:
    {
      "index": <int>,
      "type": "SE" or "PE" ...,
    }
    """
    try:
        idx = int(payload.get("index"))
        chosen_type = str(payload.get("type", "")).strip().upper()
        if chosen_type not in ALLOWED_TYPES:
            return JSONResponse(status_code=400, content={"error": "Invalid type", "allowed": ALLOWED_TYPES})

        with open("non_functional_requirements.json", "r", encoding="utf-8") as f:
            nfrs = json.load(f)

        if idx < 0 or idx >= len(nfrs):
            return JSONResponse(status_code=400, content={"error": "Invalid index"})

        item = nfrs[idx]
        desc = item.get("description", "")

        # ✅ Level via LLM ONLY (after user confirms type)
        level = predict_level_with_llm(desc, chosen_type)

        # Save to dataset (append)
        append_to_dataset(chosen_type, desc, level)

        return JSONResponse(status_code=200, content={
            "ok": True,
            "index": idx,
            "saved": {"Type": chosen_type, "Level": level, "Requirement": desc}
        })

    except Exception as e:
        logger.exception("confirm_nfr failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "confirm_nfr failed", "exception": str(e)})


@app.post("/upload_srs/")
async def upload_srs(file: UploadFile = File(...)):
    """
    Accept a single PDF file, extract requirements, and run all architecture recommendation methods
    """
    # Clean uploads folder
    try:
        for old_file in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, old_file))
            except Exception:
                pass
    except Exception:
        pass

    safe_path = os.path.join(UPLOAD_DIR, "SRS.pdf")
    try:
        with open(safe_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.exception("Failed saving uploaded file: %s", e)
        return JSONResponse(status_code=500, content={"error": "Failed to save uploaded file", "exception": str(e)})

    # Extract text
    try:
        full_text = extract_text_from_pdf(safe_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to extract text from PDF", "exception": str(e)})

    text = full_text[:MAX_CHARS]

    # ============================================================
    # Extract Functional + Non-Functional (LLM)
    # ============================================================
    nfr_prompt = f"""
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
      "description": "<rewrite the requirement professionally and assign the correct modal verb based on importance:
- Use MUST for critical requirements where failure causes system breakdown, security breach, data loss, or legal non-compliance.
- Use SHALL for mandatory requirements that the system is required to fulfill but are not life-critical.
- Use SHOULD for recommended requirements that improve quality but are not strictly mandatory.
- Use MAY for optional or nice-to-have requirements.

Your choice must reflect the real importance of the requirement based on its meaning in the SRS.>",
      "source": {{ "page": <page_number_if_known_or_null>, "start_index": <character_index_or_null> }}
    }}
  ]
}}

SRS Text:
{text}
"""
    try:
        messages = [
            {"role": "system", "content": "Return ONLY the JSON object — no explanation, no extra text, no code fences."},
            {"role": "user", "content": nfr_prompt},
        ]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=4000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        output_text = response.choices[0].message["content"]
    except Exception as e:
        logger.exception("NFR Model call failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "NFR Model request failed", "exception": str(e)})

    # Parse JSON
    try:
        json_text = extract_json_from_model_output(output_text)
        parsed = json.loads(json_text)

        with open("requirements_detailed.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        with open("functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("functional", []), f, indent=2, ensure_ascii=False)
        with open("non_functional_requirements.json", "w", encoding="utf-8") as f:
            json.dump(parsed.get("non_functional", []), f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.exception("Failed to parse model output: %s", e)
        with open("requirements_raw.txt", "w", encoding="utf-8") as f:
            f.write(output_text or "")
        return JSONResponse(status_code=500, content={"error": "Failed to parse JSON", "exception": str(e)})

    # ============================================================
    # NEW: Build low-confidence list for UI (Type only)
    # ============================================================
    low_confidence_nfrs = []
    auto_typed_nfrs = []

    try:
        tokenizer_t, model_t, le_t = load_type_model_and_encoder()
        nfrs = parsed.get("non_functional", [])

        for i, item in enumerate(nfrs):
            desc = item.get("description", "")
            pred = predict_type_with_topk(tokenizer_t, model_t, le_t, desc, top_k=3)

            best_type = pred["best_type"]
            best_conf = pred["best_confidence"]
            top = pred["top"]

            # If low confidence -> ask user
            if best_conf < TYPE_CONFIDENCE_THRESHOLD:
                low_confidence_nfrs.append({
                    "index": i,
                    "title": item.get("title", ""),
                    "description": desc,
                    "top": top,  # list of {type, confidence}
                    "best": {"type": best_type, "confidence": best_conf}
                })
            else:
                auto_typed_nfrs.append({
                    "index": i,
                    "type": best_type,
                    "confidence": best_conf
                })

    except Exception as e:
        logger.exception("Type confidence building failed: %s", e)

    # ============================================================
    # Extract Functional Requirements (chunked) — keep as-is
    # ============================================================
    logger.info("Starting functional requirements extraction...")
    chunks = chunk_text(full_text)
    all_functionals = []

    for i, chunk in enumerate(chunks):
        func_prompt = f"""
You are an expert SRS analyst. Extract ALL FUNCTIONAL REQUIREMENTS verbatim from this text.
Return ONLY a valid JSON array with no extra text, explanations, or markdown.

Format:
[{{"title": "short title", "description": "full requirement sentence", "source": {{"page": null, "start_index": null}}}}]

If no functional requirements found, return an empty array: []

Text:
{chunk}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=2500,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON array. No markdown, no explanation."},
                    {"role": "user", "content": func_prompt}
                ]
            )
            func_output = response.choices[0].message["content"]
            func_parsed = parse_model_json(func_output)
            if func_parsed:
                all_functionals.extend(func_parsed)
        except Exception as e:
            logger.warning(f"Functional extraction failed for chunk {i}: {e}")
            continue

    # Remove duplicates
    unique_functionals = []
    seen = set()
    for f in all_functionals:
        desc = f.get("description", "").strip()
        if desc and desc not in seen:
            seen.add(desc)
            unique_functionals.append(f)

    with open("extracted_functional.json", "w", encoding="utf-8") as f:
        json.dump(unique_functionals, f, indent=2, ensure_ascii=False)

    # Functional architecture method
    if not unique_functionals:
        functional_arch = {
            "chosen_architectures": [
                {
                    "key": "layered_monolith",
                    "name": "Layered / Monolithic Architecture",
                    "rationale": "Default recommendation due to insufficient functional requirements data."
                }
            ],
            "explanation": {
                "chosen_architectures": [
                    {
                        "key": "layered_monolith",
                        "name": "Layered / Monolithic Architecture",
                        "rationale": "Default recommendation due to insufficient functional requirements data."
                    }
                ],
                "scoring": {"layered_monolith": 1},
                "supporting_papers": [
                    "Taylor et al., Software Architecture: Foundations (2009)",
                    "Bass et al., Software Architecture in Practice"
                ],
                "note": "Functional requirements extraction yielded no results. Manual review recommended."
            }
        }
    else:
        functional_arch = choose_architecture(unique_functionals)

    with open("architecture_decision.json", "w", encoding="utf-8") as f:
        json.dump(functional_arch, f, indent=2, ensure_ascii=False)

    # Run methods
    ordinal_results = run_ordinal_method()
    binary_results = run_binary_method()
    weighted_results = run_weighted_score_method()
    hybrid_results = hybrid_aggregation(functional_arch, ordinal_results, binary_results, weighted_results)

    # Response
    parsed["extracted_functional_requirements"] = unique_functionals
    parsed["low_confidence_nfrs"] = low_confidence_nfrs
    parsed["auto_typed_nfrs"] = auto_typed_nfrs
    parsed["architecture_recommendations"] = {
        "functional_method": functional_arch,
        "ordinal_method": ordinal_results,
        "binary_method": binary_results,
        "weighted_score_method": weighted_results,
        "hybrid_method": hybrid_results
    }

    return JSONResponse(status_code=200, content=parsed)
