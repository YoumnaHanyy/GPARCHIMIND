from reportlab.platypus import *
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus.doctemplate import SimpleDocTemplate
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

import os


def generate_last_report(
    project,
    frs,
    nfrs,
    hybrid,
    phase4
):

    os.makedirs("data/outputs", exist_ok=True)

    pdf_path = "data/outputs/final_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=25
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=18,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=20,
        spaceAfter=12
    )

    normal_style = styles["BodyText"]

    elements = []

    # ======================================================
    # COVER
    # ======================================================

    project_name = project.get(
        "project_name",
        "Unknown Project"
    )

    elements.append(
        Spacer(1, 80)
    )

    elements.append(
        Paragraph(
            "ArchiMind Final Report",
            title_style
        )
    )

    elements.append(
        Spacer(1, 40)
    )

    elements.append(
        Paragraph(
            f"<b>Project Name:</b> {project_name}",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 60)
    )

    # ======================================================
    # FUNCTIONAL REQUIREMENTS
    # ======================================================

    elements.append(
        Paragraph(
            "Functional Requirements",
            heading_style
        )
    )

    for idx, fr in enumerate(frs, start=1):

        desc = fr.get(
            "description",
            "No description"
        )

        elements.append(
            Paragraph(
                f"<b>FR-{idx}</b>: {desc}",
                normal_style
            )
        )

        elements.append(
            Spacer(1, 8)
        )

    # ======================================================
    # NON FUNCTIONAL REQUIREMENTS
    # ======================================================

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "Non Functional Requirements",
            heading_style
        )
    )

    for idx, nfr in enumerate(nfrs, start=1):

        nfr_type = nfr.get(
            "predicted_type",
            "Unknown"
        )

        desc = nfr.get(
            "description",
            "No description"
        )

        elements.append(
            Paragraph(
                f"<b>{idx}. {nfr_type}</b>: {desc}",
                normal_style
            )
        )

        elements.append(
            Spacer(1, 8)
        )

    # ======================================================
    # SELECTED ARCHITECTURE
    # ======================================================

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Selected Architecture",
            heading_style
        )
    )

    selected_arch = hybrid.get(
        "selected_architecture",
        "Unknown"
    )

    elements.append(
        Paragraph(
            f"""
            <font size=14>
            <b>{selected_arch}</b>
            </font>
            """,
            normal_style
        )
    )

    # ======================================================
    # DESIGN PATTERNS
    # ======================================================

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Recommended Design Patterns",
            heading_style
        )
    )

    patterns = phase4.get(
        "top_patterns",
        []
    )

    if patterns:

        for idx, pattern in enumerate(patterns, start=1):

            pattern_name = pattern.get(
                "pattern",
                "Unknown Pattern"
            )

            reasons = pattern.get(
                "reasons",
                []
            )

            reason_text = ", ".join(reasons)

            elements.append(
                Paragraph(
                    f"""
                    <b>{idx}. {pattern_name}</b><br/>
                    {reason_text}
                    """,
                    normal_style
                )
            )

            elements.append(
                Spacer(1, 10)
            )

    # ======================================================
    # SUMMARY
    # ======================================================

    elements.append(
        Spacer(1, 30)
    )

    elements.append(
        Paragraph(
            "Project Summary",
            heading_style
        )
    )

    summary_text = f"""
    This report summarizes the architectural analysis,
    requirement extraction, architecture recommendation,
    and design pattern generation process for the project
    <b>{project_name}</b> using the ArchiMind platform.
    """

    elements.append(
        Paragraph(
            summary_text,
            normal_style
        )
    )

    # ======================================================
    # BUILD PDF
    # ======================================================

    doc.build(elements)

    return pdf_path