from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_verification_pdf(verification_result: dict):
    """
    Generates a formal Verification Report PDF covering:
    - Correctness
    - Completeness
    - Consistency
    """

    status = verification_result["status"]

    if status == "VERIFIED":
        path = "output/architecture_verification_report.pdf"
        title = "Architecture Verification Report"
        intro = (
            "This report documents the verification of the software architecture "
            "description. Verification ensures correctness, completeness, and "
            "consistency of the ADL prior to architectural validation."
        )
    else:
        path = "output/architecture_verification_problems.pdf"
        title = "Architecture Verification Problems Report"
        intro = (
            "This report documents verification failures detected in the "
            "architecture description. Validation was not performed because "
            "the ADL is not fully verified."
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
    # 1. Introduction
    # ==================================================
    story.append(Paragraph("<b>1. Verification Overview</b>", styles["Heading2"]))
    story.append(Paragraph(intro, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Verification Scope</b><br/>"
        "Verification evaluates the architectural description independently "
        "of quality attributes. It focuses on structural correctness, architectural "
        "completeness, and internal consistency.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # ==================================================
    # VERIFIED CASE
    # ==================================================
    if status == "VERIFIED":
        story.append(Paragraph("<b>2. Verification Result</b>", styles["Heading2"]))
        story.append(Paragraph(
            "The architecture description successfully passed all verification "
            "layers. The ADL is correct, complete, and internally consistent, "
            "and is therefore eligible for architectural validation.",
            styles["Normal"]
        ))
        doc.build(story)
        return path

    # ==================================================
    # FAILED CASE
    # ==================================================
    failed_layer = verification_result.get("failed_layer")
    details = verification_result.get("details", {})

    story.append(Paragraph("<b>2. Verification Result</b>", styles["Heading2"]))
    story.append(Paragraph(
        f"Verification Status: <b>NOT VERIFIED</b><br/>"
        f"Failed Layer: <b>{failed_layer.capitalize()}</b>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # ==================================================
    # 3. Failed Layer Analysis
    # ==================================================
    story.append(Paragraph(
        f"<b>3. {failed_layer.capitalize()} Verification Analysis</b>",
        styles["Heading2"]
    ))

    if failed_layer == "correctness":
        story.append(Paragraph(
            "Correctness verification ensures that the architecture description "
            "is syntactically and semantically well-formed. The following violations "
            "indicate malformed architectural constructs.",
            styles["Normal"]
        ))

        violations = details.get("violations", [])
        table_data = [["Rule", "Description"]] + [
            [v.get("rule"), v.get("message")] for v in violations
        ]

    elif failed_layer == "completeness":
        story.append(Paragraph(
            "Completeness verification ensures that the architecture sufficiently "
            "represents the intended system and its interactions. The following "
            "issues indicate missing or insufficient architectural coverage.",
            styles["Normal"]
        ))

        issues = details.get("issues", [])
        table_data = [["Rule", "Description"]] + [
            [i.get("rule"), i.get("message")] for i in issues
        ]

    else:  # consistency
        story.append(Paragraph(
            "Consistency verification ensures that architectural elements, "
            "metrics, and declared styles do not contradict each other. The "
            "following issues indicate internal architectural inconsistencies.",
            styles["Normal"]
        ))

        issues = details.get("issues", [])
        table_data = [["Rule", "Description"]] + [
            [i.get("rule"), i.get("message")] for i in issues
        ]

    story.append(Spacer(1, 8))

    table = Table(table_data, colWidths=[150, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    # ==================================================
    # 4. Verification Conclusion
    # ==================================================
    story.append(Paragraph("<b>4. Verification Conclusion</b>", styles["Heading2"]))
    story.append(Paragraph(
        "Due to the verification failures reported above, the architecture "
        "description cannot be considered reliable for validation. The detected "
        "issues must be resolved before quality and domain-level validation "
        "can be meaningfully performed.",
        styles["Normal"]
    ))

    doc.build(story)
    return path
