"""
Utility functions for encoding / decoding NFR labels
Used by training and inference modules
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd


# ============================================================
# 1️⃣ Load & Fit Encoders from CSV
# ============================================================

def load_type_level_encoders(csv_path: str):
    """
    Fit LabelEncoders for NFR Type and Level from training CSV

    Returns:
        le_type, le_level
    """
    df = pd.read_csv(csv_path)

    if "Type" not in df.columns or "Level" not in df.columns:
        raise ValueError("CSV must contain Type and Level columns")

    le_type = LabelEncoder()
    le_level = LabelEncoder()

    le_type.fit(df["Type"])
    le_level.fit(df["Level"])

    return le_type, le_level


# ============================================================
# 2️⃣ Encode helpers
# ============================================================

def encode_type(le_type: LabelEncoder, value: str) -> int:
    return le_type.transform([value])[0]


def encode_level(le_level: LabelEncoder, value: str) -> int:
    return le_level.transform([value])[0]


# ============================================================
# 3️⃣ Decode helpers
# ============================================================

def decode_type(le_type: LabelEncoder, idx: int) -> str:
    return le_type.inverse_transform([idx])[0]


def decode_level(le_level: LabelEncoder, idx: int) -> str:
    return le_level.inverse_transform([idx])[0]


# ============================================================
# 4️⃣ Safe decode (no crash)
# ============================================================

def safe_decode(le, idx, default="UNKNOWN"):
    try:
        return le.inverse_transform([idx])[0]
    except Exception:
        return default
