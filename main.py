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
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# =========================
# Config
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archimind")


HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CHARS = 12000
CHUNK_SIZE = 4000

# Confidence threshold for asking user
TYPE_CONF_THRESHOLD = 0.65
TOPK_SUGGESTIONS = 5

# Save user feedback here (for future training)
FEEDBACK_CSV = "user_feedback_dataset.csv"
CONFIRMED_TYPES_JSON = os.path.join(UPLOAD_DIR, "confirmed_types.json")

app = FastAPI(title="ArchiMind SRS Processor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Hugging Face LLM client ---
client = InferenceClient(model=MODEL_NAME, token=HF_API_KEY, timeout=120)

# =========================
# Helpers
# =========================
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks


def extract_json_from_model_output(output: str) -> str:
    if output is None:
        raise ValueError("Model returned empty output")

    cleaned = output.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        # maybe array
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
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_stack.append("{")
            elif ch == "}":
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack:
                        candidate = cleaned[start:i + 1]
                        json.loads(candidate)
                        return candidate

    raise ValueError("No balanced JSON object found")


# =========================
# Load TYPE model (BERT) once
# =========================
TYPE_MODEL_DIR = "./trained_nfr_type_model"
TYPE_TRAIN_CSV = "merged_NFR_cleaned_no_dots.csv"

_type_tokenizer = None
_type_model = None
_le_type = None
_type_labels = None


def load_type_model_once():
    global _type_tokenizer, _type_model, _le_type, _type_labels

    if _type_tokenizer is not None and _type_model is not None and _le_type is not None:
        return

    if not os.path.exists(TYPE_MODEL_DIR):
        raise FileNotFoundError(f"Missing type model folder: {TYPE_MODEL_DIR}")
    if not os.path.exists(TYPE_TRAIN_CSV):
        raise FileNotFoundError(f"Missing training CSV for encoders: {TYPE_TRAIN_CSV}")

    _type_tokenizer = BertTokenizer.from_pretrained(TYPE_MODEL_DIR)
    _type_model = BertForSequenceClassification.from_pretrained(TYPE_MODEL_DIR)
    _type_model.eval()

    df = pd.read_csv(TYPE_TRAIN_CSV)
    _le_type = LabelEncoder()
    _le_type.fit(df["Type"].astype(str))
    _type_labels = list(_le_type.classes_)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def predict_type_with_confidence(texts: List[str], topk: int = 5):
    """
    Returns list of dicts:
    {
      "predicted": "SE",
      "confidence": 0.77,
      "top_k": [{"label":"SE","confidence":0.77}, ...]
    }
    """
    load_type_model_once()

    tokens = _type_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        out = _type_model(**tokens)
        logits = out.logits.cpu().numpy()

    results = []
    for i in range(len(texts)):
        probs = softmax_np(logits[i])
        idx_sorted = np.argsort(probs)[::-1]
        top = idx_sorted[:topk]

        pred_idx = int(top[0])
        pred_label = _le_type.inverse_transform([pred_idx])[0]
        pred_conf = float(probs[pred_idx])

        top_k_list = []
        for j in top:
            lab = _le_type.inverse_transform([int(j)])[0]
            top_k_list.append({"label": lab, "confidence": float(probs[int(j)])})

        results.append({
            "predicted": pred_label,
            "confidence": pred_conf,
            "top_k": top_k_list
        })

    return results


def get_allowed_levels_from_arch_dataset(arch_csv="ArchitectureDataset.csv") -> List[str]:
    if not os.path.exists(arch_csv):
        return ["High", "Medium", "Low"]
    df = pd.read_csv(arch_csv)
    if "Level" not in df.columns:
        return ["High", "Medium", "Low"]
    levels = sorted(list(set(df["Level"].astype(str).dropna().tolist())))
    return levels or ["High", "Medium", "Low"]


def llm_predict_level_for_nfr(description: str, nfr_type: str, allowed_levels: List[str]) -> str:
    """
    LLM chooses ONE level from allowed_levels ONLY.
    Output JSON: {"level": "<one of allowed_levels>"}
    """
    prompt = f"""
You are an expert software architect.

Given ONE Non-Functional Requirement (NFR), choose the best matching LEVEL from this allowed list ONLY:
{allowed_levels}

NFR Type: {nfr_type}
NFR Description:
{description}

Rules:
- You MUST return ONLY valid JSON.
- Choose EXACTLY ONE level from the allowed list (case-sensitive if possible).
- If uncertain, choose the closest reasonable level from the list.

Return JSON:
{{ "level": "<one_allowed_level>" }}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return ONLY JSON. No explanation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    txt = resp.choices[0].message["content"]
    js = json.loads(extract_json_from_model_output(txt))
    level = str(js.get("level", "")).strip()

    if level not in allowed_levels:
        # fallback: pick first
        return allowed_levels[0]
    return level


def save_user_feedback_rows(rows: List[Dict[str, Any]]):
    """
    Save confirmed labels to FEEDBACK_CSV
    Schema: timestamp, title, description, type
    """
    if not rows:
        return

    out_rows = []
    ts = datetime.utcnow().isoformat()
    for r in rows:
        out_rows.append({
            "timestamp": ts,
            "title": r.get("title", ""),
            "description": r.get("description", ""),
            "Type": r.get("type", "")
        })

    df_new = pd.DataFrame(out_rows)

    if os.path.exists(FEEDBACK_CSV):
        df_old = pd.read_csv(FEEDBACK_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(FEEDBACK_CSV, index=False)
    else:
        df_new.to_csv(FEEDBACK_CSV, index=False)


def load_confirmed_types() -> Dict[str, str]:
    """
    Returns mapping: description -> confirmed_type
    """
    if not os.path.exists(CONFIRMED_TYPES_JSON):
        return {}
    try:
        with open(CONFIRMED_TYPES_JSON, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_confirmed_types(mapping: Dict[str, str]):
    with open(CONFIRMED_TYPES_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


# =========================
# Architecture Methods (same as your pipeline)
# =========================
def choose_architecture(functional_reqs: List[Dict]) -> Dict:
    text = " ".join([f"{r.get('title','')} {r.get('description','')}" for r in functional_reqs]).lower()
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
    }
    return {"chosen_architectures": chosen_architectures, "explanation": explanation}


def run_ordinal_method_with_llm_level(nfr_json_path="non_functional_requirements.json"):
    """
    Ordinal:
    - Type: from confirmed_types.json (user override) else BERT
    - Level: from LLM (NOT BERT)
    """
    try:
        # Load NFR JSON
        with open(nfr_json_path, "r", encoding="utf-8") as f:
            nfrs = json.load(f)

        confirmed_map = load_confirmed_types()
        allowed_levels = get_allowed_levels_from_arch_dataset("ArchitectureDataset.csv")

        # Predict type for all (BERT) to use if not confirmed
        texts = [item["description"] for item in nfrs]
        type_preds = predict_type_with_confidence(texts, topk=TOPK_SUGGESTIONS)

        results = []
        for i, item in enumerate(nfrs):
            desc = item["description"]
            # type (confirmed > bert)
            final_type = confirmed_map.get(desc, type_preds[i]["predicted"])
            # level (LLM)
            final_level = llm_predict_level_for_nfr(desc, final_type, allowed_levels)

            results.append({
                "title": item.get("title"),
                "description": desc,
                "predicted_type": final_type,
                "predicted_level": final_level
            })

        with open("nfr_predictions_type_level.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Recommend architectures
        pred_df = pd.DataFrame(results).rename(columns={
            "predicted_type": "Type",
            "predicted_level": "Level"
        })

        arch_df = pd.read_csv("ArchitectureDataset.csv")
        matches = pred_df.merge(arch_df, on=["Type", "Level"], how="inner")
        if "Architecture" not in matches.columns:
            # try common alternative names
            for c in ["architecture_style", "architecture style", "ArchitectureStyle"]:
                if c in matches.columns:
                    matches = matches.rename(columns={c: "Architecture"})
                    break

        style_scores = matches["Architecture"].value_counts()

        top_arch = [
            {"Architecture": style, "MatchedNFRs": int(score)}
            for style, score in style_scores.head(5).items()
        ]

        with open("Ordinal_Method_Top_Arch.json", "w", encoding="utf-8") as f:
            json.dump(top_arch, f, indent=2, ensure_ascii=False)

        return top_arch
    except Exception as e:
        logger.exception(f"Ordinal method failed: {e}")
        return []


def run_binary_method(nfr_json_path="non_functional_requirements.json"):
    try:
        tokenizer = BertTokenizer.from_pretrained("./trained_nfr_binary_model")
        model = BertForSequenceClassification.from_pretrained("./trained_nfr_binary_model")
        model.eval()

        arch_df = pd.read_csv("architecture_datasetBinary (1).csv")
        NFR_ORDER = ["PE", "SC", "MN", "A", "SE", "US", "PO", "O"]

        def predict_bin(text):
            t = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                o = model(**t)
            return int(torch.argmax(o.logits).item())

        with open(nfr_json_path, "r", encoding="utf-8") as f:
            extracted = json.load(f)

        srs = [x["description"] for x in extracted]
        srs_vector = {k: 0 for k in NFR_ORDER}

        for sentence in srs:
            for nfr_cat in NFR_ORDER:
                if nfr_cat.lower() in sentence.lower():
                    srs_vector[nfr_cat] = predict_bin(sentence)

        results = []
        for _, row in arch_df.iterrows():
            arch = row["Architecture Style"]
            arch_vec = row[NFR_ORDER].values.astype(int)
            srs_vec = np.array([srs_vector[k] for k in NFR_ORDER], dtype=int)
            diff = np.sum(np.abs(arch_vec - srs_vec))
            score = 1 - (diff / len(NFR_ORDER))
            results.append((arch, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

        output = {
            "srs_vector": srs_vector,
            "top_5_architectures": results[:5],
            "best_architecture": results[0] if results else None
        }

        with open("Binary_method_Top_arch.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        return results[:5]
    except Exception as e:
        logger.exception(f"Binary method failed: {e}")
        return []


def normalize_scores(score_dict):
    if not score_dict:
        return {}
    max_val = max(score_dict.values()) or 1
    return {k: v / max_val for k, v in score_dict.items()}


def run_weighted_score_method(nfr_json_path="non_functional_requirements.json"):
    """
    unchanged logic (as you had) – uses trained_nfr_model (type only)
    """
    try:
        MODEL_DIR = "./trained_nfr_model"
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()

        df = pd.read_csv("merged_NFR_cleaned.csv")
        le = LabelEncoder()
        le.fit(df["Type"])
        NFR_CATEGORIES = list(le.classes_)

        def predict_type(sentence):
            t = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                o = model(**t)
                pred = int(torch.argmax(o.logits, dim=1).item())
            return le.inverse_transform([pred])[0]

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
            extracted = json.load(f)

        freq_counts = {cat: 0 for cat in NFR_CATEGORIES}
        must_scores = {cat: 0.0 for cat in NFR_CATEGORIES}
        predicted = []

        for item in extracted:
            desc = item["description"]
            nfr_type = predict_type(desc)
            predicted.append(nfr_type)
            freq_counts[nfr_type] += 1
            must_scores[nfr_type] += requirement_strength(desc)

        existing = set(predicted)
        freq_counts = {k: v for k, v in freq_counts.items() if k in existing}
        must_scores = {k: v for k, v in must_scores.items() if k in existing}

        max_freq = max(freq_counts.values()) or 1
        freq_norm = {k: v / max_freq for k, v in freq_counts.items()}

        max_must = max(must_scores.values()) or 1
        must_norm = {k: v / max_must for k, v in must_scores.items()}

        # importance from full pdf sentences (same)
        pdf_path = os.path.join(UPLOAD_DIR, "SRS.pdf")
        import PyPDF2
        import nltk
        nltk.download("punkt", quiet=True)

        with open(pdf_path, "rb") as fpdf:
            reader = PyPDF2.PdfReader(fpdf)
            full = ""
            for p in reader.pages:
                full += (p.extract_text() or "") + "\n"

        sentences = nltk.sent_tokenize(full)
        counts = {t: 0 for t in NFR_CATEGORIES}
        for s in sentences:
            n = predict_type(s)
            counts[n] += 1

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
            level_norm = float(row.get("LevelNorm", row["LevelNorm"]))
            arch_scores[arch] = arch_scores.get(arch, 0) + level_norm * total_weight[nfr]

        arch_total = sum(arch_scores.values()) or 1
        arch_scores = {k: round(v / arch_total, 4) for k, v in arch_scores.items()}

        top_arch = sorted(
            [{"Architecture": k, "Score": v} for k, v in arch_scores.items()],
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
        s_f = raw_f / max_f if max_f else 0

        s_o = next((a["MatchedNFRs"] for a in ordinal if a["Architecture"] == arch), 0)
        s_b = next((score for name, score in binary if name == arch), 0)

        raw_w = next((a["Score"] for a in weighted.get("top_architectures", []) if a["Architecture"] == arch), 0)
        s_w = raw_w / max_weighted if max_weighted else 0

        final_scores[arch] = (
            0.20 * s_f +
            0.25 * s_o +
            0.20 * s_b +
            0.35 * s_w
        )

    final_scores = normalize_scores(final_scores)

    return sorted(
        [{"Architecture": k, "FinalScore": round(v * 100, 2)} for k, v in final_scores.items()],
        key=lambda x: x["FinalScore"],
        reverse=True
    )[:5]


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_srs/")
async def upload_srs(file: UploadFile = File(...)):
    """
    Phase 1:
    - Extract requirements
    - Predict NFR types + confidence
    - Return pending_questions if low confidence
    - DO NOT run architecture recs if pending exists
    """
    # clean uploads
    for old in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, old))
        except Exception:
            pass

    # reset confirmed map
    save_confirmed_types({})

    safe_path = os.path.join(UPLOAD_DIR, "SRS.pdf")
    with open(safe_path, "wb") as f:
        f.write(await file.read())

    full_text = extract_text_from_pdf(safe_path)
    text = full_text[:MAX_CHARS]

    # --- Extract FR + NFR via LLM ---
    prompt = f"""
You are an expert software analyst.
Extract both Functional and Non-Functional Requirements from the SRS text below.

Return ONLY a single clean JSON object with this exact structure:

{{
  "functional": [
    {{
      "title": "<exact title as it appears in the SRS (do not modify)>",
      "description": "<exact sentence(s) copied verbatim from the SRS (no changes)>",
      "source": {{ "page": null, "start_index": null }}
    }}
  ],
  "non_functional": [
    {{
      "title": "<exact title as it appears in the SRS (do not modify)>",
      "description": "<rewrite professionally using MUST/SHALL/SHOULD/MAY based on importance>",
      "source": {{ "page": null, "start_index": null }}
    }}
  ]
}}

Rules:
- Do not invent requirements.
- Output valid JSON only.

SRS Text:
{text}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY JSON object."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        out_text = resp.choices[0].message["content"]
        parsed = json.loads(extract_json_from_model_output(out_text))
    except Exception as e:
        logger.exception("Extraction failed")
        return JSONResponse(status_code=500, content={"error": "Extraction failed", "exception": str(e)})

    # Save extraction outputs
    with open("requirements_detailed.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    with open("functional_requirements.json", "w", encoding="utf-8") as f:
        json.dump(parsed.get("functional", []), f, indent=2, ensure_ascii=False)
    with open("non_functional_requirements.json", "w", encoding="utf-8") as f:
        json.dump(parsed.get("non_functional", []), f, indent=2, ensure_ascii=False)

    nfrs = parsed.get("non_functional", []) or []

    # --- Predict NFR type + confidence ---
    pending_questions = []
    predicted_types = []

    if nfrs:
        texts = [x.get("description", "") for x in nfrs]
        preds = predict_type_with_confidence(texts, topk=TOPK_SUGGESTIONS)

        for i, item in enumerate(nfrs):
            pred = preds[i]
            predicted_types.append({
                "index": i,
                "title": item.get("title"),
                "description": item.get("description"),
                "predicted_type": pred["predicted"],
                "confidence": pred["confidence"],
                "top_k": pred["top_k"]
            })

            if pred["confidence"] < TYPE_CONF_THRESHOLD:
                pending_questions.append({
                    "index": i,
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "suggestions": pred["top_k"],  # dropdown
                    "confidence": pred["confidence"]
                })

    # IMPORTANT: if pending exists -> stop here (no arch methods)
    response_payload = {
        "functional": parsed.get("functional", []),
        "non_functional": parsed.get("non_functional", []),
        "nfr_type_predictions": predicted_types,
        "pending_questions": pending_questions,
        "ready_for_recommendations": (len(pending_questions) == 0)
    }

    return JSONResponse(status_code=200, content=response_payload)


@app.post("/confirm_nfr/")
async def confirm_nfr(request: Request):
    """
    Phase 2a:
    Receive user's confirmed types for low-confidence NFRs.
    Save them to:
    - uploads/confirmed_types.json (for current run)
    - user_feedback_dataset.csv (for future training)
    """
    body = await request.json()
    items = body.get("items", [])

    if not isinstance(items, list):
        return JSONResponse(status_code=400, content={"error": "items must be a list"})

    confirmed_map = load_confirmed_types()

    feedback_rows = []
    for it in items:
        desc = (it.get("description") or "").strip()
        t = (it.get("type") or "").strip()

        if not desc or not t:
            continue

        confirmed_map[desc] = t
        feedback_rows.append({
            "title": it.get("title", ""),
            "description": desc,
            "type": t
        })

    save_confirmed_types(confirmed_map)
    save_user_feedback_rows(feedback_rows)

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "saved_count": len(feedback_rows),
        "saved_to": [CONFIRMED_TYPES_JSON, FEEDBACK_CSV]
    })


@app.post("/generate_recommendations/")
async def generate_recommendations():
    """
    Phase 2b:
    Run architecture recommendations ONLY after user confirmed pending NFR types.
    - Ordinal: Type (BERT/confirmed) + Level (LLM)
    - Binary: same
    - Weighted: same
    - Hybrid: same
    """
    # Load extracted requirements
    if not os.path.exists("functional_requirements.json") or not os.path.exists("non_functional_requirements.json"):
        return JSONResponse(status_code=400, content={"error": "No extracted requirements found. Upload first."})

    with open("functional_requirements.json", "r", encoding="utf-8") as f:
        functional_list = json.load(f) or []

    with open("non_functional_requirements.json", "r", encoding="utf-8") as f:
        nfr_list = json.load(f) or []

    # --- Functional method ---
    if not functional_list:
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
                "note": "Functional requirements extraction yielded no results. Manual review recommended."
            }
        }
    else:
        functional_arch = choose_architecture(functional_list)

    with open("architecture_decision.json", "w", encoding="utf-8") as f:
        json.dump(functional_arch, f, indent=2, ensure_ascii=False)

    # --- NFR-based methods ---
    ordinal_results = run_ordinal_method_with_llm_level("non_functional_requirements.json")
    binary_results = run_binary_method("non_functional_requirements.json")
    weighted_results = run_weighted_score_method("non_functional_requirements.json")

    hybrid_results = hybrid_aggregation(
        functional_arch,
        ordinal_results,
        binary_results,
        weighted_results
    )

    # Final payload
    final_payload = {
        "functional": functional_list,
        "non_functional": nfr_list,
        "architecture_recommendations": {
            "functional_method": functional_arch,
            "ordinal_method": ordinal_results,
            "binary_method": binary_results,
            "weighted_score_method": weighted_results,
            "hybrid_method": hybrid_results
        }
    }

    return JSONResponse(status_code=200, content=final_payload)
