"""
ArchiMind — SRS Validation Service
=====================================
Orchestrates the full validation pipeline:
  1. Validate inputs
  2. Run SRSValidator
  3. Persist results
  4. Clean up temp files

Usage:
    from service.srs_validation_service import validate_srs_pipeline

    result = validate_srs_pipeline(
        pdf_path="uploads/val_abc123.pdf",
        validation_id="val_abc123",
        original_filename="MySystem_SRS.pdf",
    )
"""

import os
import logging

from ai.validations.srs_validator import SRSValidator


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ──────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when the validation pipeline fails for a known reason."""
    pass


class EmptyDocumentError(ValidationError):
    """Raised when no text could be extracted from the PDF."""
    pass


class InsufficientContentError(ValidationError):
    """Raised when the document has too little content to validate."""
    pass


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────

def validate_srs_pipeline(
    pdf_path: str,
    validation_id: str,
    original_filename: str = "document.pdf",
) -> dict:
    """
    End-to-end SRS validation pipeline.

    Args:
        pdf_path:          Absolute or relative path to the uploaded PDF.
        validation_id:     Unique identifier for this validation run.
        original_filename: The user's original filename (for display).

    Returns:
        A dict containing all validation results ready for API response
        and MongoDB storage.

    Raises:
        EmptyDocumentError:       If no text could be extracted.
        InsufficientContentError: If the document is too short.
        ValidationError:          For any other known pipeline failures.
    """

    logger.info(
        "Starting SRS validation | id=%s | file=%s",
        validation_id,
        original_filename
    )

    # ── 1. Guard: file exists ───────────────────
    if not os.path.exists(pdf_path):
        raise ValidationError(
            f"PDF file not found at path: {pdf_path}"
        )

    # ── 2. Run validation ──────────────────────
    try:
        validator = SRSValidator()
        result = validator.validate_document(
            pdf_path=pdf_path,
            validation_id=validation_id,
        )
    except EmptyDocumentError:
        raise
    except InsufficientContentError:
        raise
    except Exception as exc:
        logger.exception(
            "Validation pipeline failed | id=%s",
            validation_id
        )
        raise ValidationError(
            f"Validation pipeline encountered an error: {exc}"
        ) from exc

    # ── 3. Post-process ────────────────────────
    result["original_filename"] = original_filename
    result["validation_id"]     = validation_id

    # Compute a human-readable grade
    score = result.get("quality_score", 0)
    result["grade"] = _score_to_grade(score)
    result["grade_label"] = _score_to_label(score)

    logger.info(
        "Validation completed | id=%s | score=%.1f | issues=%d",
        validation_id,
        score,
        len(result.get("issues", []))
    )

    # ── 4. Cleanup temp PDF ────────────────────
    _cleanup_file(pdf_path)

    return result


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _score_to_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _score_to_label(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 80: return "Good"
    if score >= 70: return "Acceptable"
    if score >= 60: return "Needs Improvement"
    return "Poor"


def _cleanup_file(path: str) -> None:
    """Remove the temporary uploaded PDF after processing."""
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug("Cleaned up temp file: %s", path)
    except OSError as e:
        logger.warning("Could not remove temp file %s: %s", path, e)