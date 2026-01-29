"""
PDF utilities:
- Read PDF text
- Split text into sentences
"""

import os
import PyPDF2
import nltk
import fitz

# ============================================================
# 1️⃣ Load PDF text
# ============================================================

def load_pdf_text(pdf_path: str) -> str:
    """
    Extract full text from a PDF file
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


# ============================================================
# 2️⃣ Split text into sentences
# ============================================================

def split_into_sentences(text: str):
    """
    Split text into sentences using NLTK
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    return nltk.sent_tokenize(text)

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text