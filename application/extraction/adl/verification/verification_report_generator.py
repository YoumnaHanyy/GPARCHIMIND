from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def generate_verification_pdf(verification_result: dict):
    """
    Generates a Verification Report PDF ONLY if verification is successful.
    """

    # ⛔ Stop immediately if verification failed
    if verification_result.get("status") != "VERIFIED":
        return None

    path = "data/outputs/architecture_verification_report.pdf"
    title = "Architecture Verification Report"
    intro = (
        "This report documents the verification of the software architecture "
        "description. Verification ensures correctness, completeness, and "
        "consistency of the ADL prior to architectural validation."
    )

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ==================================================
    # Title
    # ==================================================
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 14))

    # ==================================================
    # 1. Verification Overview
    # ==================================================
    story.append(Paragraph("<b>1. Verification Overview</b>", styles["Heading2"]))
    story.append(Paragraph(intro, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Verification Scope</b><br/>"
        "Verification evaluates the architectural description independently "
        "of quality attributes. It focuses on structural correctness, "
        "architectural completeness, and internal consistency.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # ==================================================
    # 2. Verification Result
    # ==================================================
    story.append(Paragraph("<b>2. Verification Result</b>", styles["Heading2"]))
    story.append(Paragraph(
        "The architecture description successfully passed all verification "
        "layers. The ADL is correct, complete, and internally consistent, "
        "and is therefore eligible for architectural validation.",
        styles["Normal"]
    ))

    doc.build(story)
    return path
