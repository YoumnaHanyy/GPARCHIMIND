from reportlab.platypus import *
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus.doctemplate import SimpleDocTemplate
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os

def generate_last_report(
    project,
    frs,
    nfrs,
    hybrid,
    phase4,
    generated_code=None
):
    # ======================================================
    # OUTPUT PATH
    # ======================================================
    os.makedirs("data/outputs", exist_ok=True)
    pdf_path = "data/outputs/final_report.pdf"

    # ======================================================
    # DOCUMENT SETUP
    # ======================================================
    # Usable width = 612 (letter width) - 80 (margins) = 532
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # ======================================================
    # COLOR PALETTE (Modern Black & Green Theme)
    # ======================================================
    PRIMARY_BLACK = colors.HexColor("#0f0f0f")  # Deep Black
    ACCENT_GREEN = colors.HexColor("#10b981")   # Modern Emerald Green
    TEXT_DARK = colors.HexColor("#1f2937")      # Dark Gray
    TEXT_MUTED = colors.HexColor("#6b7280")     # Muted Gray
    BG_LIGHT = colors.HexColor("#ecfdf5")       # Very Light Green Tint for tables
    BORDER_GRAY = colors.HexColor("#d1d5db")    # Light Gray for subtle borders

    # ======================================================
    # PAGE BORDER FUNCTION (New)
    # ======================================================
    def draw_page_border(canvas, doc):
        canvas.saveState()
        # Set Border Color and Thickness
        canvas.setStrokeColor(ACCENT_GREEN)
        canvas.setLineWidth(2)
        
        # Draw a rectangle slightly inside the page edges (20 points from the edge)
        page_width, page_height = letter
        margin = 20
        canvas.rect(margin, margin, page_width - (2 * margin), page_height - (2 * margin))
        
        # Optional: Add small inner corner lines for a "Tech/Scanner" vibe
        canvas.setStrokeColor(PRIMARY_BLACK)
        canvas.setLineWidth(1)
        inner_margin = 25
        canvas.rect(inner_margin, inner_margin, page_width - (2 * inner_margin), page_height - (2 * inner_margin))
        
        canvas.restoreState()

    # ======================================================
    # CUSTOM STYLES
    # ======================================================
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=PRIMARY_BLACK,
        spaceBefore=25,
        spaceAfter=15,
        borderPadding=0
    )

    def create_heading(text):
        return [
            Paragraph(text, heading_style),
            Table([['']], colWidths=[532], rowHeights=[2], style=[('BACKGROUND', (0,0), (0,0), ACCENT_GREEN)]),
            Spacer(1, 15)
        ]

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY
    )
    
    table_cell_style = ParagraphStyle(
        "TableCellStyle",
        parent=normal_style,
        fontSize=10,
        leading=13
    )

    highlight_style = ParagraphStyle(
        "HighlightStyle",
        parent=normal_style,
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=PRIMARY_BLACK,
        backColor=BG_LIGHT,
        alignment=TA_CENTER,
        borderPadding=12,
        spaceBefore=10,
        spaceAfter=10,
        borderRadius=4,
        borderColor=ACCENT_GREEN,
        borderWidth=1.5
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=ACCENT_GREEN,  
        backColor=PRIMARY_BLACK, 
        borderPadding=12,
        borderColor=ACCENT_GREEN,
        borderWidth=1,
        borderRadius=4,
        spaceBefore=10,
        spaceAfter=10
    )

    def get_grid_style():
        return TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PRIMARY_BLACK),
            ('TEXTCOLOR', (0,0), (-1,0), ACCENT_GREEN),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('TOPPADDING', (0,0), (-1,0), 12),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('PADDING', (0,1), (-1,-1), 8),
        ])

    elements = []

    # ======================================================
    # 🌟 CREATIVE COVER PAGE
    # ======================================================
    project_name = project.get("project_name", "Unknown Project")
    current_date = datetime.now().strftime("%B %d, %Y")

    logo_path = os.path.join("presentation", "static", "images", "Logo_black.png")
    
    if os.path.exists(logo_path):
        try:
            img_reader = ImageReader(logo_path)
            img_w, img_h = img_reader.getSize()
            target_width = 130
            target_height = img_h * (target_width / img_w)
            
            logo = Image(logo_path, width=target_width, height=target_height)
            logo.hAlign = 'LEFT' 
            elements.append(logo)
        except Exception as e:
            pass
    
    elements.append(Spacer(1, 40))

    elements.append(Table([['']], colWidths=[100], rowHeights=[6], style=[('BACKGROUND', (0,0), (0,0), ACCENT_GREEN)], hAlign='LEFT'))
    elements.append(Spacer(1, 20))

    hero_title = ParagraphStyle(
        "HeroTitle",
        fontName="Helvetica-Bold",
        fontSize=50,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    elements.append(Paragraph('<font color="#10b981">Archi</font><font color="#0f0f0f">Mind</font>', hero_title))

    hero_subtitle = ParagraphStyle(
        "HeroSubTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=TEXT_MUTED,
        alignment=TA_LEFT,
        spaceAfter=70
    )
    elements.append(Paragraph("E N T E R P R I S E   A R C H I T E C T U R E   R E P O R T", hero_subtitle))

    details_data = [
        ["PROJECT NAME", project_name.upper()],
        ["GENERATED ON", current_date],
        ["GENERATED BY", "ArchiMind AI Engine"],
        ["STATUS", "Finalized Blueprint"]
    ]

    details_table = Table(details_data, colWidths=[140, 300], hAlign='LEFT')
    details_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('TEXTCOLOR', (0,0), (0,-1), ACCENT_GREEN),
        ('TEXTCOLOR', (1,0), (1,-1), PRIMARY_BLACK),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, BORDER_GRAY),
    ]))
    
    elements.append(details_table)

    elements.append(Spacer(1, 100))

    footer_note_style = ParagraphStyle(
        "FooterNote",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=TEXT_MUTED,
        alignment=TA_LEFT,
        leading=13
    )
    elements.append(Paragraph(
        "<b>CONFIDENTIAL:</b> This document contains proprietary architecture analysis, recommendations, <br/>"
        "and AI-generated design models tailored specifically for this project.",
        footer_note_style
    ))

    elements.append(PageBreak())

    # ======================================================
    # PROJECT SUMMARY
    # ======================================================
    elements.extend(create_heading("Project Summary"))
    summary_text = f"""
    This report summarizes the complete software architecture engineering 
    workflow performed using ArchiMind for the project <b>{project_name}</b>.<br/><br/>
    The platform extracted and analyzed functional and non-functional requirements, 
    generated architecture recommendations using hybrid AI-based evaluation methods, 
    recommended suitable design patterns, and generated an initial code skeleton 
    based on the selected architecture style.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 20))

    # ======================================================
    # FUNCTIONAL REQUIREMENTS
    # ======================================================
    elements.extend(create_heading("Functional Requirements (FRs)"))
    
    if frs:
        fr_data = [["ID", "Description"]]
        for idx, fr in enumerate(frs, start=1):
            desc = fr.get("description", "No description")
            fr_data.append([f"FR-{idx}", Paragraph(desc, table_cell_style)])
        
        t_fr = Table(fr_data, colWidths=[60, 472])
        t_fr.setStyle(get_grid_style())
        elements.append(t_fr)
    else:
        elements.append(Paragraph("No functional requirements found.", normal_style))
        
    elements.append(Spacer(1, 20))

    # ======================================================
    # NON-FUNCTIONAL REQUIREMENTS
    # ======================================================
    elements.extend(create_heading("Non-Functional Requirements (NFRs)"))
    
    if nfrs:
        nfr_data = [["#", "Type", "Description"]]
        for idx, nfr in enumerate(nfrs, start=1):
            nfr_type = nfr.get("predicted_type", "Unknown")
            desc = nfr.get("description", "No description")
            nfr_data.append([str(idx), nfr_type, Paragraph(desc, table_cell_style)])
            
        t_nfr = Table(nfr_data, colWidths=[30, 100, 402])
        t_nfr.setStyle(get_grid_style())
        elements.append(t_nfr)
    else:
        elements.append(Paragraph("No non-functional requirements found.", normal_style))

    elements.append(Spacer(1, 20))

    # ======================================================
    # TOP HYBRID ARCHITECTURES
    # ======================================================
    elements.extend(create_heading("Top Recommended Architectures"))
    
    top_archs = hybrid.get("top_architectures", [])
    if top_archs:
        arch_data = [["Rank", "Architecture Style", "Match Score"]]
        for idx, arch in enumerate(top_archs, start=1):
            arch_name = arch.get("Architecture", "Unknown")
            score = arch.get("FinalScore", 0)
            arch_data.append([str(idx), arch_name, f"{score}%"])
            
        t_arch = Table(arch_data, colWidths=[50, 350, 132])
        t_arch.setStyle(get_grid_style())
        elements.append(t_arch)

    elements.append(Spacer(1, 20))

    # ======================================================
    # SELECTED ARCHITECTURE
    # ======================================================
    selected_arch = hybrid.get("selected_architecture", "Unknown")
    elements.extend(create_heading("Selected Architecture"))
    elements.append(Paragraph(f"<b>{selected_arch}</b>", highlight_style))

    elements.append(Spacer(1, 20))

    # ======================================================
    # DESIGN PATTERNS
    # ======================================================
    elements.extend(create_heading("Recommended Design Patterns"))
    
    patterns = phase4.get("top_patterns", [])
    if patterns:
        pat_data = [["#", "Pattern", "Justification"]]
        for idx, pattern in enumerate(patterns, start=1):
            pattern_name = pattern.get("pattern", "Unknown Pattern")
            reasons = pattern.get("reasons", [])
            reason_text = " • " + "<br/> • ".join(reasons) if reasons else "No reasons provided."
            pat_data.append([str(idx), pattern_name, Paragraph(reason_text, table_cell_style)])
            
        t_pat = Table(pat_data, colWidths=[30, 130, 372])
        t_pat.setStyle(get_grid_style())
        elements.append(t_pat)

    elements.append(Spacer(1, 20))

    # ======================================================
    # ADL REPORT SECTION
    # ======================================================

    elements.extend(
     create_heading(
        "Architecture Blueprint (ADL)"
    )
   )

    elements.append(
    Paragraph(
        """
        The Architecture Description Language
        (ADL) blueprint was successfully
        generated, verified, and validated
        using the ArchiMind platform.
        """,
        normal_style
    )
)

    elements.append(
    Spacer(1, 15)
)

# ======================================================
# DOWNLOAD LINKS TABLE
# ======================================================

    project_id = project.get(
    "project_id",
    ""
)

    base_url = "http://127.0.0.1:8000"

    adl_link = \
    f"{base_url}/api/report/{project_id}"

    verification_link = \
    f"{base_url}/download-verification-report/{project_id}"

    validation_link = \
    f"{base_url}/download-validation-report"

    links_data = [

    [
        Paragraph(
            "<b>Document</b>",
            table_cell_style
        ),

        Paragraph(
            "<b>Download Link</b>",
            table_cell_style
        )
    ],

    [
        "ADL Blueprint",

        Paragraph(
            f'''
            <link href="{adl_link}">
            Download ADL Report
            </link>
            ''',
            table_cell_style
        )
    ],

    [
        "Verification Report",

        Paragraph(
            f'''
            <link href="{verification_link}">
            Download Verification Report
            </link>
            ''',
            table_cell_style
        )
    ],

    [
        "Validation Report",

        Paragraph(
            f'''
            <link href="{validation_link}">
            Download Validation Report
            </link>
            ''',
            table_cell_style
        )
    ]
]

    links_table = Table(

     links_data,

    colWidths=[180, 352]
    )

    links_table.setStyle(
    get_grid_style()
    )

    elements.append(
    links_table
    )


    # ======================================================
    # CODE SKELETON
    # ======================================================
    elements.append(PageBreak())
    elements.extend(create_heading("Generated Code Skeleton"))

    if generated_code:
        safe_code = generated_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        elements.append(Paragraph(safe_code, code_style))
    else:
        elements.append(Paragraph("No code skeleton generated.", normal_style))

    # ======================================================
    # BUILD PDF (With the Border!)
    # ======================================================
    # ضفنا الميزتين دول عشان ينفذوا دالة الـ Border في أول صفحة وباقي الصفحات
    doc.build(
        elements, 
        onFirstPage=draw_page_border, 
        onLaterPages=draw_page_border
    )

    return pdf_path