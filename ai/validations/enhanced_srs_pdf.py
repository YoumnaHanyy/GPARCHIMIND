"""
ArchiMind — Enhanced SRS PDF Generator
=======================================
Generates the original SRS document structure as a PDF,
with enhanced (fixed) requirements highlighted in green
and original problematic ones shown with red background.

PDF Structure:
  Cover Page
  Legend
  ──────────
  [Original SRS sections in order]
    • Normal text      → rendered as-is
    • Fixed requirement → green ✦ ENHANCED card
    • Normal requirement → gray card
  ──────────
  Appendix A: Compliance Scores
  Appendix B: Detected Issues Summary
  Appendix C: Improvement Suggestions
"""

import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT


# ── COLORS ──────────────────────────────────
GREEN        = colors.HexColor("#0e7e59")
GREEN_LIGHT  = colors.HexColor("#d1fae5")
GREEN_DARK   = colors.HexColor("#064e3b")
CYAN         = colors.HexColor("#06b6d4")
RED          = colors.HexColor("#dc2626")
RED_LIGHT    = colors.HexColor("#fee2e2")
ORANGE       = colors.HexColor("#d97706")
ORANGE_LIGHT = colors.HexColor("#fef3c7")
BLUE         = colors.HexColor("#1d4ed8")
GRAY         = colors.HexColor("#6b7280")
GRAY_LIGHT   = colors.HexColor("#f9fafb")
GRAY_LINE    = colors.HexColor("#e5e7eb")
DARK         = colors.HexColor("#1f2937")
WHITE        = colors.white

SEV_COLORS = {
    "Critical": (colors.HexColor("#4c0519"), colors.HexColor("#fecdd3")),
    "High":     (RED,    RED_LIGHT),
    "Medium":   (ORANGE, ORANGE_LIGHT),
    "Low":      (GREEN,  GREEN_LIGHT),
}


# ── STYLES ──────────────────────────────────

def _styles() -> dict:
    return {
        "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold",
            fontSize=28, textColor=WHITE, alignment=TA_CENTER, spaceAfter=6),
        "cover_sub": ParagraphStyle("cover_sub", fontName="Helvetica",
            fontSize=12, textColor=colors.HexColor("#a7f3d0"),
            alignment=TA_CENTER, spaceAfter=4),
        "cover_meta": ParagraphStyle("cover_meta", fontName="Helvetica",
            fontSize=9, textColor=colors.HexColor("#6ee7b7"), alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14,
            textColor=GREEN_DARK, spaceBefore=18, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
            textColor=DARK, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
            textColor=DARK, leading=16, spaceAfter=6, alignment=TA_JUSTIFY),
        "body_sm": ParagraphStyle("body_sm", fontName="Helvetica", fontSize=9,
            textColor=GRAY, leading=14, spaceAfter=4),
        "req_normal": ParagraphStyle("req_normal", fontName="Courier", fontSize=9,
            textColor=DARK, leading=14, spaceAfter=4),
        "req_enhanced": ParagraphStyle("req_enhanced", fontName="Courier-Bold",
            fontSize=10, textColor=GREEN_DARK, leading=15, spaceAfter=4),
        "issues_fixed": ParagraphStyle("issues_fixed", fontName="Helvetica",
            fontSize=9, textColor=colors.HexColor("#065f46"), leading=13),
        "rationale": ParagraphStyle("rationale", fontName="Helvetica-Oblique",
            fontSize=9, textColor=colors.HexColor("#78350f"), leading=13),
        "suggestion": ParagraphStyle("suggestion", fontName="Helvetica",
            fontSize=10, textColor=DARK, leading=15, leftIndent=18, spaceAfter=6),
        "score_label": ParagraphStyle("score_label", fontName="Helvetica-Bold",
            fontSize=10, textColor=DARK),
        "score_val": ParagraphStyle("score_val", fontName="Helvetica-Bold",
            fontSize=10, textColor=GREEN, alignment=TA_CENTER),
    }


# ── HELPERS ─────────────────────────────────

def _divider(title: str, S: dict) -> list:
    return [
        Spacer(1, 0.3 * cm),
        Paragraph(title, S["h1"]),
        HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=8),
    ]


def _compliance_bar(label: str, value: float, S: dict) -> Table:
    bar_w  = 9 * cm
    filled = bar_w * (value / 100)
    empty  = bar_w - filled
    color  = GREEN if value >= 80 else ORANGE if value >= 60 else RED

    inner = Table([["", ""]], colWidths=[filled, empty], rowHeights=[10])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), color),
        ("BACKGROUND", (1, 0), (1, 0), GRAY_LINE),
    ]))
    row = Table(
        [[Paragraph(label, S["score_label"]), inner,
          Paragraph(f"{value:.0f}%", S["score_val"])]],
        colWidths=[4 * cm, bar_w, 2 * cm], rowHeights=[22],
    )
    row.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return row


# ── CARDS ───────────────────────────────────

def _enhanced_card(req_id, original, enhanced, issues_fixed, rationale, S) -> KeepTogether:
    """Green card: original (red bg) + enhanced (green bg)."""

    header = Table([[
        Paragraph(f"<b>{req_id}</b>", ParagraphStyle(
            "hid", fontName="Courier-Bold", fontSize=10, textColor=WHITE)),
        Paragraph("✦  ENHANCED", ParagraphStyle(
            "henh", fontName="Helvetica-Bold", fontSize=8,
            textColor=colors.HexColor("#a7f3d0"), alignment=TA_RIGHT)),
    ]], colWidths=[4*cm, 11*cm])
    header.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))

    parts = []

    # Original — red background
    if original:
        orig = original[:220] + ("..." if len(original) > 220 else "")
        orig_t = Table(
            [[Paragraph(f"<font color='#7f1d1d'><b>Original:</b>  {orig}</font>",
                        S["body_sm"])]],
            colWidths=[15*cm])
        orig_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), RED_LIGHT),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ]))
        parts.append(orig_t)

    # Enhanced — green background
    enh_rows = [[Paragraph(
        f"<font color='#064e3b'><b>Enhanced:</b>  {enhanced}</font>",
        S["req_enhanced"])]]
    if issues_fixed:
        enh_rows.append([Paragraph(
            f"<font color='#065f46'>✓  <b>Issues Fixed:</b>  {issues_fixed}</font>",
            S["issues_fixed"])])
    if rationale:
        enh_rows.append([Paragraph(
            f"<font color='#78350f'><i>Why:  {rationale}</i></font>",
            S["rationale"])])

    enh_t = Table(enh_rows, colWidths=[15*cm])
    enh_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN_LIGHT),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("BOX",           (0,0),(-1,-1), 1.5, GREEN),
    ]))
    parts.append(enh_t)

    return KeepTogether([header] + parts + [Spacer(1, 0.3*cm)])


def _normal_req_card(text: str, S: dict) -> Table:
    t = Table([[Paragraph(text, S["req_normal"])]], colWidths=[15*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GRAY_LIGHT),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("BOX",           (0,0),(-1,-1), 0.5, GRAY_LINE),
    ]))
    return t


# ── PARSERS ─────────────────────────────────

def _parse_enhanced(enhanced_srs: str) -> dict:
    """Parse enhanced_srs text → { req_id: {enhanced, issues_fixed, rationale} }"""
    result = {}
    for block in enhanced_srs.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines    = block.split("\n")
        id_match = re.match(r"^\[([^\]]+)\]\s*(.*)", lines[0].strip())
        if not id_match:
            continue
        req_id = id_match.group(1).strip()
        enh    = id_match.group(2).strip()
        fixed  = ""
        why    = ""
        for line in lines[1:]:
            line = line.strip()
            if "Issues Fixed:" in line:
                fixed = line.replace("✦ Issues Fixed:", "").strip()
            elif "Rationale:" in line:
                why = line.replace("✦ Rationale:", "").strip()
        result[req_id] = {"enhanced": enh, "issues_fixed": fixed, "rationale": why}
    return result


def _parse_sections(text: str) -> list:
    """Split raw SRS text into [{heading, content}] preserving order."""
    lines   = text.split("\n")
    heading = "Document"
    buf     = []
    sections = []

    heading_re = re.compile(
        r"^(\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-/]{3,70}$"
    )

    for line in lines:
        s = line.strip()
        is_heading = (
            s and len(s) < 80
            and heading_re.match(s)
            and not s.endswith(".")
            and not re.search(r'\b(shall|must|will|should|may)\b', s, re.I)
        )
        if is_heading and buf:
            sections.append({"heading": heading, "content": "\n".join(buf).strip()})
            heading = s
            buf     = []
        else:
            buf.append(line)

    if buf:
        sections.append({"heading": heading, "content": "\n".join(buf).strip()})

    return [s for s in sections if s["content"].strip()]


# ── MAIN ────────────────────────────────────

def generate_enhanced_srs_pdf(validation_data: dict) -> bytes:
    buffer = io.BytesIO()
    S      = _styles()

    project_name  = validation_data.get("project_name",        "Unknown Project")
    quality_score = validation_data.get("quality_score",       0)
    grade         = validation_data.get("grade",               "N/A")
    grade_label   = validation_data.get("grade_label",         "")
    issues        = validation_data.get("issues",              [])
    compliance    = validation_data.get("compliance",          {})
    suggestions   = validation_data.get("suggestions",         [])
    enhanced_srs  = validation_data.get("enhanced_srs",        "")
    ai_summary    = validation_data.get("ai_summary",          "")
    original_text = validation_data.get("original_text",       "")
    orig_reqs     = validation_data.get("original_requirements", [])
    generated_at  = datetime.utcnow().strftime("%d %B %Y, %H:%M UTC")

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
        title=f"Enhanced SRS — {project_name}",
        author="ArchiMind",
    )

    story      = []
    enh_map    = _parse_enhanced(enhanced_srs)
    total_enhs = len(enh_map)

    # ── Match original requirement → enhanced version ──
    def _find_enhanced(req_text: str):
        for req_id, data in enh_map.items():
            idx_match = re.search(r"(\d+)$", req_id)
            if idx_match:
                idx = int(idx_match.group(1)) - 1
                if 0 <= idx < len(orig_reqs):
                    if orig_reqs[idx][:60] in req_text or req_text[:60] in orig_reqs[idx]:
                        return req_id, data
            # fuzzy fallback
            if req_text[:50].lower() in data["enhanced"].lower():
                return req_id, data
        return None, None

    # ════════════════════════════════════════
    # COVER
    # ════════════════════════════════════════
    cover_inner = [
        [Paragraph("ArchiMind  ·  SRS Validation Studio", S["cover_sub"])],
        [Spacer(1, 0.3*cm)],
        [Paragraph("Enhanced SRS Report", S["cover_title"])],
        [Paragraph(project_name, ParagraphStyle("pn", fontName="Helvetica-Bold",
            fontSize=17, textColor=WHITE, alignment=TA_CENTER, spaceAfter=6))],
        [Spacer(1, 0.7*cm)],
        [Table([[
            Paragraph(f"<b>{quality_score:.0f}%</b>", ParagraphStyle("sc",
                fontName="Helvetica-Bold", fontSize=28, textColor=WHITE,
                alignment=TA_CENTER)),
            Paragraph(f"<b>{grade}</b>", ParagraphStyle("gr",
                fontName="Helvetica-Bold", fontSize=28,
                textColor=colors.HexColor("#a7f3d0"), alignment=TA_CENTER)),
            Paragraph(f"<b>{total_enhs}</b>", ParagraphStyle("er",
                fontName="Helvetica-Bold", fontSize=28, textColor=WHITE,
                alignment=TA_CENTER)),
        ],[
            Paragraph("Quality Score", S["cover_meta"]),
            Paragraph(grade_label,     S["cover_meta"]),
            Paragraph("Reqs Enhanced", S["cover_meta"]),
        ]], colWidths=[5*cm, 5*cm, 5*cm])],
        [Spacer(1, 0.7*cm)],
        [Paragraph(f"Generated: {generated_at}  |  IEEE 830 / ISO 25010",
                   S["cover_meta"])],
    ]
    cover = Table(cover_inner, colWidths=[15*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 28),
        ("BOTTOMPADDING", (0,0),(-1,-1), 28),
        ("LEFTPADDING",   (0,0),(-1,-1), 24),
        ("RIGHTPADDING",  (0,0),(-1,-1), 24),
        ("ROUNDEDCORNERS",[14,14,14,14]),
    ]))
    story.append(cover)

    # Legend
    story.append(Spacer(1, 0.5*cm))
    leg = Table([[
        Table([[[Paragraph("  ✦ ENHANCED", ParagraphStyle("le",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=WHITE, alignment=TA_CENTER))]]], colWidths=[3.5*cm]),
        Paragraph("Requirement rewritten by AI to fix detected issues", S["body_sm"]),
        Spacer(0.4*cm, 0),
        Table([[[Paragraph("  Original", ParagraphStyle("lo",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=colors.HexColor("#7f1d1d"), alignment=TA_CENTER))]]], colWidths=[3*cm]),
        Paragraph("Original problematic text (shown for reference)", S["body_sm"]),
    ]], colWidths=[3.5*cm, 6.5*cm, 0.4*cm, 3*cm, 7*cm])
    leg.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), GREEN_DARK),
        ("BACKGROUND", (3,0),(3,0), RED_LIGHT),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(leg)
    story.append(PageBreak())

    # ════════════════════════════════════════
    # ORIGINAL SRS — section by section
    # ════════════════════════════════════════
    story.append(Paragraph(
        "The following sections reproduce the original SRS in order. "
        "Requirements with detected issues are shown with a "
        "<font color='#0e7e59'><b>green ✦ ENHANCED</b></font> card "
        "containing the fixed version.",
        S["body"],
    ))
    story.append(Spacer(1, 0.4*cm))

    if original_text:
        sections = _parse_sections(original_text)
        for section in sections:
            story += _divider(section["heading"], S)

            # Split into chunks (paragraphs or lines)
            chunks = [c.strip() for c in re.split(r"\n{2,}", section["content"]) if c.strip()]
            if not chunks:
                chunks = [l.strip() for l in section["content"].split("\n") if l.strip()]

            for chunk in chunks:
                has_req = bool(re.search(
                    r'\b(shall|must|will|should)\b', chunk, re.I
                ))
                if has_req:
                    req_id, enh_data = _find_enhanced(chunk)
                    if req_id and enh_data:
                        story.append(_enhanced_card(
                            req_id       = req_id,
                            original     = chunk,
                            enhanced     = enh_data["enhanced"],
                            issues_fixed = enh_data["issues_fixed"],
                            rationale    = enh_data["rationale"],
                            S            = S,
                        ))
                    else:
                        story.append(_normal_req_card(chunk, S))
                        story.append(Spacer(1, 0.15*cm))
                else:
                    story.append(Paragraph(chunk, S["body"]))
    else:
        story.append(Paragraph(
            "Original SRS text not available. Re-run validation to regenerate.",
            S["body"],
        ))

    # ════════════════════════════════════════
    # APPENDIX A — COMPLIANCE
    # ════════════════════════════════════════
    story.append(PageBreak())
    story += _divider("Appendix A — Compliance Analysis", S)

    for label, key in [
        ("Clarity",      "clarity"),
        ("Completeness", "completeness"),
        ("Consistency",  "consistency"),
        ("Testability",  "testability"),
        ("Traceability", "traceability"),
    ]:
        story.append(_compliance_bar(label, compliance.get(key, 0), S))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"<b>Overall: {quality_score:.1f}/100  —  Grade: {grade} ({grade_label})</b>",
        S["h2"],
    ))

    if ai_summary and not ai_summary.startswith("AI deep analysis unavailable"):
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>AI Executive Summary</b>", S["h2"]))
        story.append(Paragraph(ai_summary, S["body"]))

    # ════════════════════════════════════════
    # APPENDIX B — ISSUES
    # ════════════════════════════════════════
    story.append(PageBreak())
    story += _divider(f"Appendix B — Detected Issues  ({len(issues)} total)", S)

    if not issues:
        story.append(Paragraph("No issues detected.", S["body"]))
    else:
        grouped: dict = {}
        for iss in issues:
            cat = iss.get("category", "General")
            grouped.setdefault(cat, []).append(iss)

        for cat, cat_issues in grouped.items():
            story.append(Paragraph(f"<b>{cat}</b>  ({len(cat_issues)})", S["h2"]))
            for iss in cat_issues:
                sev    = iss.get("severity", "Low")
                fg, bg = SEV_COLORS.get(sev, (GRAY, GRAY_LIGHT))
                hdr = Table([[
                    Paragraph(f"<b>{iss.get('req_id','?')}</b>", ParagraphStyle(
                        "rid", fontName="Helvetica-Bold", fontSize=9, textColor=fg)),
                    Paragraph(f"<b>{sev}</b>", ParagraphStyle(
                        "rsev", fontName="Helvetica-Bold", fontSize=8,
                        textColor=fg, alignment=TA_RIGHT)),
                ]], colWidths=[10*cm, 5*cm])
                hdr.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0),(-1,-1), bg),
                    ("TOPPADDING",    (0,0),(-1,-1), 5),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                    ("LEFTPADDING",   (0,0),(-1,-1), 8),
                    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                ]))
                brows = [
                    [Paragraph(f"<b>Problem:</b> {iss.get('problem','')}", S["body_sm"])],
                    [Paragraph(f"<b>Suggestion:</b> {iss.get('suggestion','')}", S["body_sm"])],
                ]
                if iss.get("rule_ref"):
                    brows.append([Paragraph(
                        f"<font color='#1d4ed8'><b>Rule:</b> {iss['rule_ref']}</font>",
                        S["body_sm"])])
                bdy = Table(brows, colWidths=[15*cm])
                bdy.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0),(-1,-1), WHITE),
                    ("LEFTPADDING",   (0,0),(-1,-1), 8),
                    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                    ("TOPPADDING",    (0,0),(-1,-1), 4),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                    ("BOX",           (0,0),(-1,-1), 0.5, GRAY_LINE),
                ]))
                story.append(KeepTogether([hdr, bdy, Spacer(1, 0.2*cm)]))

    # ════════════════════════════════════════
    # APPENDIX C — SUGGESTIONS
    # ════════════════════════════════════════
    story += _divider("Appendix C — Improvement Suggestions", S)
    if not suggestions:
        story.append(Paragraph("No suggestions generated.", S["body"]))
    else:
        for i, s in enumerate(suggestions, 1):
            story.append(Paragraph(f"<b>{i}.</b>  {s}", S["suggestion"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()