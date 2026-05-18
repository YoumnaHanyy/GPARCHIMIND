"""
ArchiMind — Enhanced SRS PDF Generator  (v3 — Modified Sections Only)
======================================================================
Instead of reproducing the full SRS, this generator outputs ONLY the
sections that contain at least one enhanced requirement.

Per-section layout:
  ┌─────────────────────────────────────────────┐
  │  Section heading  │  X issues  │  X enhanced │
  ├─────────────────────────────────────────────┤
  │  [For each enhanced requirement in section] │
  │    ● BEFORE card  (red)                     │
  │    ● AFTER card   (green) + issues + why    │
  └─────────────────────────────────────────────┘

Document outline:
  1. Cover Page
  2. Executive Summary  (score + compliance bars + AI summary)
  3. Modified Sections  (one block per section, only changed reqs)
  4. Appendix A — Full Issue List
  5. Appendix B — Improvement Suggestions
"""

import io
import re
from datetime import datetime
from collections import defaultdict

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

# ── PALETTE ─────────────────────────────────────────────────────────────────
GREEN        = colors.HexColor("#059669")
GREEN_LIGHT  = colors.HexColor("#d1fae5")
GREEN_DARK   = colors.HexColor("#064e3b")
GREEN_MID    = colors.HexColor("#065f46")
RED          = colors.HexColor("#dc2626")
RED_LIGHT    = colors.HexColor("#fff1f2")
RED_DARK     = colors.HexColor("#7f1d1d")
ORANGE       = colors.HexColor("#d97706")
ORANGE_LIGHT = colors.HexColor("#fffbeb")
BLUE         = colors.HexColor("#1d4ed8")
BLUE_LIGHT   = colors.HexColor("#eff6ff")
GRAY         = colors.HexColor("#6b7280")
GRAY_LIGHT   = colors.HexColor("#f8fafc")
GRAY_LINE    = colors.HexColor("#e2e8f0")
GRAY_MID     = colors.HexColor("#94a3b8")
DARK         = colors.HexColor("#0f172a")
SLATE        = colors.HexColor("#334155")
WHITE        = colors.white
SECTION_BG   = colors.HexColor("#f0fdf4")
DIVIDER_COL  = colors.HexColor("#bbf7d0")

PAGE_W = A4[0] - 4 * cm   # usable width with 2cm margins each side

SEV_COLORS = {
    "Critical": (colors.HexColor("#4c0519"), colors.HexColor("#ffe4e6")),
    "High":     (RED,    colors.HexColor("#fee2e2")),
    "Medium":   (ORANGE, ORANGE_LIGHT),
    "Low":      (GREEN,  GREEN_LIGHT),
}


# ── STYLES ───────────────────────────────────────────────────────────────────

def _styles() -> dict:
    return {
        "cover_brand": ParagraphStyle("cover_brand",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=colors.HexColor("#6ee7b7"), alignment=TA_CENTER, spaceAfter=4),
        "cover_title": ParagraphStyle("cover_title",
            fontName="Helvetica-Bold", fontSize=26,
            textColor=WHITE, alignment=TA_CENTER, spaceAfter=4),
        "cover_project": ParagraphStyle("cover_project",
            fontName="Helvetica-Bold", fontSize=15,
            textColor=colors.HexColor("#a7f3d0"), alignment=TA_CENTER, spaceAfter=4),
        "cover_meta": ParagraphStyle("cover_meta",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#6ee7b7"), alignment=TA_CENTER),
        "cover_stat_val": ParagraphStyle("cover_stat_val",
            fontName="Helvetica-Bold", fontSize=30,
            textColor=WHITE, alignment=TA_CENTER),
        "cover_stat_lbl": ParagraphStyle("cover_stat_lbl",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#6ee7b7"), alignment=TA_CENTER),
        "h1": ParagraphStyle("h1",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=GREEN_DARK, spaceBefore=14, spaceAfter=4),
        "h2": ParagraphStyle("h2",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=SLATE, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=10,
            textColor=DARK, leading=16, spaceAfter=5, alignment=TA_JUSTIFY),
        "body_sm": ParagraphStyle("body_sm",
            fontName="Helvetica", fontSize=9,
            textColor=SLATE, leading=14, spaceAfter=3),
        "body_xs": ParagraphStyle("body_xs",
            fontName="Helvetica", fontSize=8,
            textColor=GRAY, leading=12),
        "req_before": ParagraphStyle("req_before",
            fontName="Courier", fontSize=9,
            textColor=RED_DARK, leading=14),
        "req_after": ParagraphStyle("req_after",
            fontName="Courier-Bold", fontSize=9,
            textColor=GREEN_DARK, leading=14),
        "issue_inline": ParagraphStyle("issue_inline",
            fontName="Helvetica", fontSize=8.5,
            textColor=GREEN_MID, leading=12),
        "rationale_inline": ParagraphStyle("rationale_inline",
            fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=colors.HexColor("#78350f"), leading=12),
        "sec_title": ParagraphStyle("sec_title",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=GREEN_DARK),
        "sec_badge": ParagraphStyle("sec_badge",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=WHITE, alignment=TA_CENTER),
        "score_label": ParagraphStyle("score_label",
            fontName="Helvetica-Bold", fontSize=9, textColor=SLATE),
        "score_val": ParagraphStyle("score_val",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=GREEN, alignment=TA_RIGHT),
        "suggestion": ParagraphStyle("suggestion",
            fontName="Helvetica", fontSize=10,
            textColor=DARK, leading=15, leftIndent=14, spaceAfter=5),
        "app_heading": ParagraphStyle("app_heading",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK, spaceBefore=14, spaceAfter=5),
    }


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _hr(color=GRAY_LINE, thickness=1) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6)


def _spacer(h=0.25) -> Spacer:
    return Spacer(1, h * cm)


def _compliance_bar(label: str, value: float, S: dict) -> Table:
    bar_w  = 8 * cm
    filled = max(bar_w * (value / 100), 0.01 * cm)
    empty  = bar_w - filled
    col    = GREEN if value >= 80 else ORANGE if value >= 60 else RED

    bar_inner = Table([["", ""]], colWidths=[filled, empty], rowHeights=[8])
    bar_inner.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), col),
        ("BACKGROUND", (1,0),(1,0), GRAY_LINE),
    ]))
    row = Table(
        [[Paragraph(label, S["score_label"]), bar_inner,
          Paragraph(f"{value:.0f}%", S["score_val"])]],
        colWidths=[4*cm, bar_w, 1.5*cm], rowHeights=[20],
    )
    row.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0),(-1,-1), 2),
        ("RIGHTPADDING",(0,0),(-1,-1), 2),
    ]))
    return row


# ── SECTION HEADER ───────────────────────────────────────────────────────────

def _section_header(section_name: str, n_issues: int, n_enhanced: int, S: dict) -> list:
    title_cell = Paragraph(f"§  {section_name}", S["sec_title"])

    issues_badge = Table(
        [[Paragraph(f"{n_issues}  issues", S["sec_badge"])]],
        colWidths=[2.4*cm], rowHeights=[18],
    )
    issues_badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), RED),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("ROUNDEDCORNERS",[6,6,6,6]),
    ]))

    enh_badge = Table(
        [[Paragraph(f"{n_enhanced}  enhanced", S["sec_badge"])]],
        colWidths=[2.9*cm], rowHeights=[18],
    )
    enh_badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("ROUNDEDCORNERS",[6,6,6,6]),
    ]))

    badges_row = Table(
        [[issues_badge, _spacer(0.2), enh_badge]],
        colWidths=[2.4*cm, 0.3*cm, 2.9*cm],
    )
    badges_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))

    header_row = Table(
        [[title_cell, badges_row]],
        colWidths=[PAGE_W - 6.2*cm, 6.2*cm],
    )
    header_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), SECTION_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LINEBELOW",     (0,0),(-1,-1), 2, GREEN),
        ("ROUNDEDCORNERS",[6,6,0,0]),
    ]))

    return [_spacer(0.4), header_row]


# ── COMPARISON CARD ──────────────────────────────────────────────────────────

def _comparison_card(req_id: str, original: str, enhanced: str,
                     issues_fixed: str, rationale: str,
                     req_issues: list, S: dict) -> list:
    """
    One before/after card per requirement.
    Returns a list of flowables (splitByRow-safe, no full-card KeepTogether).
    """
    flowables = []

    # ── req ID bar ──
    req_id_bar = Table([[
        Paragraph(f"  {req_id}  ", ParagraphStyle(
            "rid_bar", fontName="Courier-Bold", fontSize=9,
            textColor=WHITE, alignment=TA_LEFT)),
        Paragraph(
            f"  {len(req_issues)} issue{'s' if len(req_issues) != 1 else ''}  ",
            ParagraphStyle("rcnt", fontName="Helvetica", fontSize=8,
                           textColor=GRAY_MID, alignment=TA_RIGHT)),
    ]], colWidths=[PAGE_W * 0.55, PAGE_W * 0.45])
    req_id_bar.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), colors.HexColor("#1e293b")),
        ("BACKGROUND",    (1,0),(1,0), colors.HexColor("#0f172a")),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))

    # ── BEFORE block ──
    before_label = Table(
        [[Paragraph("  BEFORE  ", ParagraphStyle(
            "blbl", fontName="Helvetica-Bold", fontSize=7.5,
            textColor=WHITE, alignment=TA_CENTER))]],
        colWidths=[1.6*cm], rowHeights=[14],
    )
    before_label.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), RED),
        ("TOPPADDING",    (0,0),(-1,-1), 1),
        ("BOTTOMPADDING", (0,0),(-1,-1), 1),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
    ]))

    before_header_row = Table(
        [[before_label, _spacer(0.2),
          Paragraph("Original requirement", S["body_xs"])]],
        colWidths=[1.6*cm, 0.25*cm, PAGE_W - 1.85*cm],
    )
    before_header_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))

    orig_short = original[:300] + ("…" if len(original) > 300 else "")

    before_rows = [
        [before_header_row],
        [_spacer(0.15)],
        [Paragraph(orig_short, S["req_before"])],
    ]

    if req_issues:
        issue_data = []
        for iss in req_issues:
            sev = iss.get("severity", "Low")
            fg, _ = SEV_COLORS.get(sev, (GRAY, GRAY_LIGHT))
            issue_data.append([
                Paragraph(f"● {iss.get('problem','')}", S["body_xs"]),
                Paragraph(f"[{sev}]", ParagraphStyle(
                    "sevlbl", fontName="Helvetica-Bold", fontSize=7.5,
                    textColor=fg, alignment=TA_RIGHT)),
            ])
        issues_tbl = Table(
            issue_data,
            colWidths=[PAGE_W - 2*cm - 1.8*cm, 1.8*cm],
        )
        issues_tbl.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 2),
            ("RIGHTPADDING", (0,0),(-1,-1), 2),
            ("TOPPADDING",   (0,0),(-1,-1), 1),
            ("BOTTOMPADDING",(0,0),(-1,-1), 1),
            ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ]))
        before_rows += [
            [_spacer(0.1)],
            [Paragraph("Detected issues:", ParagraphStyle(
                "dilbl", fontName="Helvetica-Bold", fontSize=8,
                textColor=RED_DARK))],
            [issues_tbl],
        ]

    before_block = Table(before_rows, colWidths=[PAGE_W - 2*cm])
    before_block.splitByRow = 1
    before_block.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), RED_LIGHT),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LINEAFTER",     (0,0),(0,-1), 3, RED),
    ]))

    # ── Connector arrow ──
    arrow = Table(
        [[Paragraph("▼", ParagraphStyle(
            "arr", fontName="Helvetica-Bold", fontSize=14,
            textColor=GREEN_MID, alignment=TA_CENTER))]],
        colWidths=[PAGE_W],
    )
    arrow.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
    ]))

    # ── AFTER block ──
    after_label = Table(
        [[Paragraph("  AFTER  ", ParagraphStyle(
            "albl", fontName="Helvetica-Bold", fontSize=7.5,
            textColor=WHITE, alignment=TA_CENTER))]],
        colWidths=[1.4*cm], rowHeights=[14],
    )
    after_label.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN),
        ("TOPPADDING",    (0,0),(-1,-1), 1),
        ("BOTTOMPADDING", (0,0),(-1,-1), 1),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
    ]))

    after_header_row = Table(
        [[after_label, _spacer(0.2),
          Paragraph("Enhanced requirement", S["body_xs"])]],
        colWidths=[1.4*cm, 0.25*cm, PAGE_W - 1.65*cm],
    )
    after_header_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))

    after_rows = [
        [after_header_row],
        [_spacer(0.15)],
        [Paragraph(enhanced, S["req_after"])],
    ]
    if issues_fixed:
        after_rows += [
            [_spacer(0.1)],
            [Paragraph(f"✓  {issues_fixed}", S["issue_inline"])],
        ]
    if rationale:
        after_rows += [
            [_spacer(0.08)],
            [Paragraph(f"Why:  {rationale}", S["rationale_inline"])],
        ]

    after_block = Table(after_rows, colWidths=[PAGE_W - 2*cm])
    after_block.splitByRow = 1
    after_block.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREEN_LIGHT),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LINEAFTER",     (0,0),(0,-1), 3, GREEN),
        ("BOX",           (0,0),(-1,-1), 1, DIVIDER_COL),
    ]))

    # Keep req_id_bar glued to the BEFORE block only (prevents orphan header)
    flowables.append(KeepTogether([req_id_bar, before_block]))
    flowables.append(arrow)
    flowables.append(after_block)
    flowables.append(_spacer(0.4))

    return flowables


# ── PARSERS ──────────────────────────────────────────────────────────────────

def _parse_enhanced(enhanced_srs: str) -> dict:
    """Parse enhanced_srs text → { req_id: {enhanced, issues_fixed, rationale, original} }"""
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
        orig   = ""
        for line in lines[1:]:
            line = line.strip()
            if "Issues Fixed:" in line:
                fixed = re.sub(r"^[✦●\-]?\s*Issues Fixed:\s*", "", line).strip()
            elif "Rationale:" in line:
                why = re.sub(r"^[✦●\-]?\s*Rationale:\s*", "", line).strip()
            elif "Original:" in line:
                orig = re.sub(r"^[✦●\-]?\s*Original:\s*", "", line).strip()
        result[req_id] = {
            "enhanced":     enh,
            "issues_fixed": fixed,
            "rationale":    why,
            "original":     orig,
        }
    return result


def _parse_sections(text: str) -> list:
    """Split raw SRS text into [{heading, content}] preserving order."""
    lines    = text.split("\n")
    heading  = "Document"
    buf      = []
    sections = []

    heading_re = re.compile(r"^(\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-/]{3,70}$")

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


def _find_enhanced(req_text: str, enh_map: dict):
    """Match a requirement chunk to its enhanced version using fuzzy overlap."""
    req_clean = req_text.strip().lower()
    for req_id, data in enh_map.items():
        original = data.get("original", "").strip().lower()
        if original and len(original) > 10:
            if original[:80] in req_clean or req_clean[:80] in original:
                return req_id, data
            orig_words = set(original.split())
            req_words  = set(req_clean.split())
            if len(orig_words) > 4 and len(req_words) > 4:
                overlap = len(orig_words & req_words) / max(len(orig_words), len(req_words))
                if overlap > 0.55:
                    return req_id, data
    return None, None


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def generate_enhanced_srs_pdf(validation_data: dict) -> bytes:
    buffer = io.BytesIO()
    S      = _styles()

    # ── Extract the real project/system name from the SRS title line ──
    # Priority: explicit "project_name" key → extracted from original_text title → fallback
    _raw_project_name = validation_data.get("project_name", "")
    _original_text_tmp = validation_data.get("original_text", "")

    def _extract_project_title(raw_name: str, orig_text: str) -> str:
        """
        Try to get the actual system/project name, not the supervisor/author name.
        Strategy:
          1. If raw_name looks like a system name (not "Dr." / "Eng." / person names), use it.
          2. Otherwise scan the first 30 lines of the SRS for a title pattern like
             "Software Requirement Specification ... for <Project>" or
             "SRS for <Project>" or just the first prominent non-person heading.
        """
        import re as _re

        # Patterns that suggest it's a person/author name, not a project title
        person_patterns = [
            r'Dr', r'Eng', r'Prof', r'Mr', r'Ms', r'Mrs',
            r'Supervised', r'Prepared', r'Submitted',
        ]
        def looks_like_person(s):
            return any(_re.search(p, s, _re.I) for p in person_patterns)

        # 1. If the stored name is clean, use it
        if raw_name and not looks_like_person(raw_name) and len(raw_name) < 80:
            return raw_name

        # 2. Scan original_text for SRS title patterns
        if orig_text:
            lines = orig_text.split("\n")[:40]
            # Pattern: "... for <Title>" or "... of <Title>"
            for_pattern = _re.compile(
    r"(?:specification|srs|document|system|requirements?)"
    r"\s+(?:for|of)\s+([A-Z][A-Za-z0-9 \-&:,']{2,60})",
    _re.I
)
            for line in lines:
                line = line.strip()
                if not line or looks_like_person(line):
                    continue
                m = for_pattern.search(line)
                if m:
                    candidate = m.group(1).strip().rstrip(".,")
                    if len(candidate) > 3:
                        return candidate
            # Fallback: first non-empty, non-person line that looks like a title
            for line in lines:
                line = line.strip()
                if (line and len(line) > 4 and len(line) < 80
                        and not looks_like_person(line)
                        and not line.startswith("#")
                        and not _re.match(r'^\d', line)):
                    return line

        return raw_name if raw_name else "Unknown Project"

    project_name = _extract_project_title(_raw_project_name, _original_text_tmp)
    quality_score = validation_data.get("quality_score",        0)
    grade         = validation_data.get("grade",                "N/A")
    grade_label   = validation_data.get("grade_label",          "")
    issues        = validation_data.get("issues",               [])
    compliance    = validation_data.get("compliance",           {})
    suggestions   = validation_data.get("suggestions",          [])
    enhanced_srs  = validation_data.get("enhanced_srs",         "")
    ai_summary    = validation_data.get("ai_summary",           "")
    original_text = validation_data.get("original_text",        "")
    generated_at  = datetime.utcnow().strftime("%d %B %Y, %H:%M UTC")

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
        title=f"ArchiMind — {project_name} — Changes Report",
        author="ArchiMind",
    )

    story   = []
    enh_map = _parse_enhanced(enhanced_srs)

    # Build issue lookup keyed by req_id
    issues_by_req: dict = defaultdict(list)
    for iss in issues:
        rid = iss.get("req_id", "")
        if rid:
            issues_by_req[rid].append(iss)

    # ══════════════════════════════════════════════════════════
    # 1. COVER PAGE
    # Two-part layout:
    #   ┌─────────────────────────────────┐
    #   │   brand · title · project name  │  ← top block (dark green)
    #   │   description · generated date  │
    #   ├─────────────────────────────────┤
    #   │  85%  │  B  │  12  │  45        │  ← stats strip (darker green)
    #   │  lbl  │ lbl │ lbl  │ lbl        │
    #   └─────────────────────────────────┘
    # ══════════════════════════════════════════════════════════

    # ── Top content block ──
    # Use raw project_name as supervisor/author credit if it looks like a person name
    import re as _re2
    _person_check = [r'\bDr\b', r'\bEng\b', r'\bProf\b', r'\bSupervised\b']
    _raw_looks_like_person = any(
        _re2.search(p, _raw_project_name, _re2.I) for p in _person_check
    ) if _raw_project_name else False

    top_content = [
        [Paragraph("ArchiMind  ·  SRS Validation Studio", S["cover_brand"])],
        [Spacer(1, 0.35 * cm)],
        [Paragraph("SRS Changes Report", S["cover_title"])],
        [Spacer(1, 0.3 * cm)],
        # Project title — the actual system name
        [Paragraph(project_name, S["cover_project"])],
    ]

    # If the original stored name was supervisor/author names, show them as a subtitle
    if _raw_looks_like_person and _raw_project_name != project_name:
        top_content += [
            [Spacer(1, 0.15 * cm)],
            [Paragraph(_raw_project_name, ParagraphStyle(
                "cover_supervisor", fontName="Helvetica", fontSize=10,
                textColor=colors.HexColor("#6ee7b7"), alignment=TA_CENTER))],
        ]

    top_content += [
        [Spacer(1, 0.5 * cm)],
        [Paragraph(
            "This report shows only requirements that were modified.<br/>"
            "Each section displays the original issues and the enhanced version.",
            ParagraphStyle("cover_desc", fontName="Helvetica", fontSize=9,
                           textColor=colors.HexColor("#a7f3d0"),
                           alignment=TA_CENTER, leading=15))],
        [Spacer(1, 0.3 * cm)],
        [Paragraph(
            f"Generated: {generated_at}  ·  IEEE 830 / ISO 25010",
            S["cover_meta"])],
    ]
    top_block = Table(top_content, colWidths=[15 * cm])
    top_block.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 36),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
        ("LEFTPADDING",   (0, 0), (-1, -1), 28),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 28),
        ("ROUNDEDCORNERS", [12, 12, 0, 0]),
    ]))

    # Stack cover block only
    story.append(top_block)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 2. EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(_hr(GREEN, 2))
    story.append(_spacer(0.1))

    for label, key in [
        ("Clarity",      "clarity"),
        ("Completeness", "completeness"),
        ("Consistency",  "consistency"),
        ("Testability",  "testability"),
        ("Traceability", "traceability"),
    ]:
        story.append(_compliance_bar(label, compliance.get(key, 0), S))
        story.append(_spacer(0.12))

    story.append(_spacer(0.1))
    story.append(Paragraph(
        f"<b>Overall Score: {quality_score:.1f} / 100  —  Grade: {grade} ({grade_label})</b>",
        S["h2"],
    ))

    if ai_summary and not ai_summary.startswith("AI deep analysis unavailable"):
        story.append(_spacer(0.2))
        story.append(Paragraph("<b>AI Analysis</b>", S["h2"]))
        story.append(Paragraph(ai_summary, S["body"]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 3. MODIFIED SECTIONS ONLY
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("Modified Requirements", S["h1"]))
    story.append(_hr(GREEN, 2))
    story.append(Paragraph(
        "Only sections containing enhanced requirements are shown below. "
        "Each requirement card shows the original version with its detected issues, "
        "followed by the AI-enhanced version with the improvement rationale.",
        S["body"],
    ))
    story.append(_spacer(0.3))

    if original_text:
        sections = _parse_sections(original_text)

        for section in sections:
            chunks = [c.strip() for c in re.split(r"\n{2,}", section["content"]) if c.strip()]
            if not chunks:
                chunks = [l.strip() for l in section["content"].split("\n") if l.strip()]

            # Collect only the enhanced reqs in this section
            section_cards      = []
            section_issue_count = 0

            for chunk in chunks:
                has_req = bool(re.search(r'\b(shall|must|will|should)\b', chunk, re.I))
                if not has_req:
                    continue
                req_id, enh_data = _find_enhanced(chunk, enh_map)
                if req_id and enh_data:
                    req_issues = issues_by_req.get(req_id, [])
                    section_issue_count += len(req_issues)
                    section_cards.append((req_id, chunk, enh_data, req_issues))

            # Skip this section entirely if nothing was changed
            if not section_cards:
                continue

            # Section header bar
            story.extend(_section_header(
                section_name = section["heading"],
                n_issues     = section_issue_count,
                n_enhanced   = len(section_cards),
                S            = S,
            ))
            story.append(_spacer(0.2))

            # One card per enhanced requirement
            for req_id, original_chunk, enh_data, req_issues in section_cards:
                story.extend(_comparison_card(
                    req_id       = req_id,
                    original     = original_chunk,
                    enhanced     = enh_data["enhanced"],
                    issues_fixed = enh_data["issues_fixed"],
                    rationale    = enh_data["rationale"],
                    req_issues   = req_issues,
                    S            = S,
                ))

    else:
        story.append(Paragraph(
            "Original SRS text not available. Re-run validation to regenerate.",
            S["body"],
        ))

    # ══════════════════════════════════════════════════════════
    # 4. APPENDIX A — FULL ISSUE LIST
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph(
        f"Appendix A — All Detected Issues  ({len(issues)} total)",
        S["app_heading"]))
    story.append(_hr(GRAY_LINE, 1))
    story.append(_spacer(0.1))

    if not issues:
        story.append(Paragraph("No issues detected.", S["body"]))
    else:
        grouped: dict = defaultdict(list)
        for iss in issues:
            grouped[iss.get("category", "General")].append(iss)

        for cat, cat_issues in grouped.items():
            story.append(Paragraph(f"<b>{cat}</b>  ({len(cat_issues)})", S["h2"]))
            for iss in cat_issues:
                sev    = iss.get("severity", "Low")
                fg, bg = SEV_COLORS.get(sev, (GRAY, GRAY_LIGHT))

                hdr = Table([[
                    Paragraph(f"<b>{iss.get('req_id','?')}</b>",
                        ParagraphStyle("rid3", fontName="Helvetica-Bold",
                                       fontSize=9, textColor=fg)),
                    Paragraph(f"<b>{sev}</b>",
                        ParagraphStyle("rsev2", fontName="Helvetica-Bold",
                                       fontSize=8, textColor=fg, alignment=TA_RIGHT)),
                ]], colWidths=[PAGE_W * 0.6, PAGE_W * 0.4])
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
                bdy = Table(brows, colWidths=[PAGE_W])
                bdy.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0),(-1,-1), WHITE),
                    ("LEFTPADDING",   (0,0),(-1,-1), 8),
                    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                    ("TOPPADDING",    (0,0),(-1,-1), 4),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                    ("BOX",           (0,0),(-1,-1), 0.5, GRAY_LINE),
                ]))
                story.append(KeepTogether([hdr, bdy, _spacer(0.2)]))

    # ══════════════════════════════════════════════════════════
    # 5. APPENDIX B — SUGGESTIONS
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Appendix B — Improvement Suggestions", S["app_heading"]))
    story.append(_hr(GRAY_LINE, 1))
    story.append(_spacer(0.1))

    if not suggestions:
        story.append(Paragraph("No suggestions generated.", S["body"]))
    else:
        for i, s in enumerate(suggestions, 1):
            story.append(Paragraph(f"<b>{i}.</b>  {s}", S["suggestion"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()