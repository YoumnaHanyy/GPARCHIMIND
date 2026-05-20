"""
Architecture Verification PDF Report Generator
"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
import os

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_NAVY     = HexColor("#1A365D")
_SLATE    = HexColor("#4A5568")
_PASS_BG  = HexColor("#C6F6D5")
_PASS_FG  = HexColor("#22543D")
_FAIL_BG  = HexColor("#FED7D7")
_FAIL_FG  = HexColor("#9B2335")
_WARN_BG  = HexColor("#FEFCBF")
_WARN_FG  = HexColor("#744210")
_THEAD    = HexColor("#2D3748")
_ROW_ALT  = HexColor("#F7FAFC")
_DIVIDER  = HexColor("#E2E8F0")
_ACCENT   = HexColor("#EBF8FF")

# ---------------------------------------------------------------------------
# Rule catalogues
# ---------------------------------------------------------------------------
CORRECTNESS_RULES = [
    ("COMPONENT_EXISTENCE",
     "At least one component must be defined in the architecture."),
    ("COMPONENT_NAMING",
     "Every component must have a non-empty name."),
    ("COMPONENT_UNIQUENESS",
     "Component names must be unique across the architecture."),
    ("UNDEFINED_SOURCE",
     "Every relationship source must reference a declared component."),
    ("UNDEFINED_TARGET",
     "Every relationship target must reference a declared component."),
    ("SELF_LOOP",
     "A relationship's source and target must not be the same component."),
    ("INVALID_RELATIONSHIP_TYPE",
     "Relationship types must belong to the supported vocabulary "
     "(event-flow, data-flow)."),
]

COMPLETENESS_RULES = [
    ("NO_RELATIONSHIPS",
     "An architecture with components must declare interactions between them."),
    ("ISOLATED_COMPONENT",
     "Every component must participate in at least one relationship."),
    ("INSUFFICIENT_COMPONENTS",
     "A microservices architecture must contain multiple services."),
]

CONSISTENCY_RULES = [
    ("MISSING_STYLE",
     "An architectural style must be declared."),
    ("NEGATIVE_METRIC",
     "Fan-in and fan-out metrics must never be negative."),
    ("CENTRALIZATION_OUT_OF_RANGE",
     "The centralization index must lie in the [0,1] interval."),
    ("STYLE_COMMUNICATION_CONFLICT",
     "A microservices architecture must not rely on synchronous communication."),
    ("STYLE_STATE_CONFLICT",
     "A microservices architecture must not exhibit high state concentration."),
]

LAYER_CATALOGUE = {
    "correctness":  ("Correctness",  CORRECTNESS_RULES,  "violations"),
    "completeness": ("Completeness", COMPLETENESS_RULES, "issues"),
    "consistency":  ("Consistency",  CONSISTENCY_RULES,  "issues"),
}

_SEVERITY = {
    "correctness":  "High",
    "completeness": "Medium",
    "consistency":  "Medium",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _layer_score(layer_key, layer_result):
    """Returns (passed, total, pct_int)."""
    _, rules, fk = LAYER_CATALOGUE[layer_key]
    total  = len(rules)
    failed = {f.get("rule") for f in layer_result.get(fk, [])}
    passed = sum(1 for rid, _ in rules if rid not in failed)
    return passed, total, int(passed / total * 100) if total else 0


def derive_suggested_fixes(verification_result: dict) -> list:
    fixes = []
    layers = verification_result.get("layers", {})

    _CORRECTNESS_MAP = {
        "COMPONENT_EXISTENCE":
            "Component Existence: No components declared. Add at least one component.",
        "COMPONENT_NAMING":
            "Component Naming: One or more components are missing a name. Assign a non-empty unique name to every component.",
        "COMPONENT_UNIQUENESS":
            "Component Uniqueness: Duplicate component names detected. Rename duplicates so every component is uniquely identifiable.",
        "UNDEFINED_SOURCE":
            "Undefined Source Reference: One or more relationship sources reference components not declared in the model.",
        "UNDEFINED_TARGET":
            "Undefined Target Reference: One or more relationship targets reference components not declared in the model.",
        "SELF_LOOP":
            "Self-Loops: Remove relationships where source and target are the same component.",
        "INVALID_RELATIONSHIP_TYPE":
            "Invalid Relationship Types: Use only 'event-flow' or 'data-flow' as relationship types.",
    }
    _COMPLETENESS_MAP = {
        "NO_RELATIONSHIPS":
            "Missing Interactions: Declare at least one relationship between components.",
        "ISOLATED_COMPONENT":
            "Isolated Components: Wire all components into the architecture or remove unused ones.",
        "INSUFFICIENT_COMPONENTS":
            "Insufficient Services: A microservices architecture must contain multiple services.",
    }
    _CONSISTENCY_MAP = {
        "MISSING_STYLE":
            "Missing Architectural Style: Specify a style (e.g. Layered, Microservices, Event-Driven).",
        "NEGATIVE_METRIC":
            "Negative Metric: A negative fan-in or fan-out indicates a graph-construction defect.",
        "CENTRALIZATION_OUT_OF_RANGE":
            "Centralization Index out of [0,1]: Check graph construction for errors.",
        "STYLE_COMMUNICATION_CONFLICT":
            "Style/Communication Conflict: Microservices must use asynchronous communication, not synchronous.",
        "STYLE_STATE_CONFLICT":
            "Style/State Conflict: Distribute state across services to conform to microservices principles.",
    }

    for v in layers.get("correctness", {}).get("violations", []):
        msg = _CORRECTNESS_MAP.get(v.get("rule"))
        if msg:
            fixes.append(msg)

    for i in layers.get("completeness", {}).get("issues", []):
        msg = _COMPLETENESS_MAP.get(i.get("rule"))
        if msg:
            fixes.append(msg)

    for i in layers.get("consistency", {}).get("issues", []):
        msg = _CONSISTENCY_MAP.get(i.get("rule"))
        if msg:
            fixes.append(msg)

    seen, out = set(), []
    for f in fixes:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# PDF paragraph styles
# ---------------------------------------------------------------------------
def _styles():
    base = getSampleStyleSheet()
    def ps(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "CoverTitle": ps("VCoverTitle", fontSize=26, fontName="Helvetica-Bold",
                         textColor=white, spaceAfter=6),
        "CoverSub":   ps("VCoverSub",   fontSize=11, fontName="Helvetica",
                         textColor=HexColor("#A0AEC0"), spaceAfter=0),
        "H2":         ps("VH2",         fontSize=14, fontName="Helvetica-Bold",
                         textColor=_NAVY, spaceBefore=14, spaceAfter=6),
        "H3":         ps("VH3",         fontSize=11, fontName="Helvetica-Bold",
                         textColor=_SLATE, spaceBefore=8, spaceAfter=4),
        "Body":       ps("VBody",       fontSize=10, fontName="Helvetica",
                         textColor=_SLATE, spaceAfter=6, leading=15),
        "Bullet":     ps("VBullet",     fontSize=10, fontName="Helvetica",
                         textColor=_SLATE, leftIndent=14, spaceAfter=4, leading=14),
        "StatusOK":   ps("VStatusOK",   fontSize=14, fontName="Helvetica-Bold",
                         textColor=_PASS_FG, spaceAfter=8),
        "StatusFail": ps("VStatusFail", fontSize=14, fontName="Helvetica-Bold",
                         textColor=_FAIL_FG, spaceAfter=8),
        "TblHdr":     ps("VTblHdr",     fontSize=9,  fontName="Helvetica-Bold",
                         textColor=white),
        "TblCell":    ps("VTblCell",    fontSize=9,  fontName="Helvetica",
                         textColor=_SLATE, leading=13),
        "TblBold":    ps("VTblBold",    fontSize=9,  fontName="Helvetica-Bold",
                         textColor=_NAVY),
    }


# ---------------------------------------------------------------------------
# Footer callback
# ---------------------------------------------------------------------------
def _footer(canvas, doc):
    w, _ = A4
    canvas.saveState()
    canvas.setStrokeColor(_DIVIDER)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_SLATE)
    canvas.drawString(2 * cm, 1.0 * cm, "ArchiMind  |  Architecture Verification Report")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Table style builder
# ---------------------------------------------------------------------------
def _base_ts(n_data_rows):
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), _THEAD),
        ("GRID",       (0, 0), (-1, -1), 0.5, _DIVIDER),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ]
    for i in range(1, n_data_rows + 1):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), _ROW_ALT))
    return ts


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _build_cover(story, st, status, failed_layers, layers):
    # Dark navy header banner
    hdr = Table(
        [[Paragraph("Architecture Verification Report", st["CoverTitle"])],
         [Paragraph("Automated Architecture Quality Assurance  |  ArchiMind AI", st["CoverSub"])]],
        colWidths=[16.5 * cm],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _NAVY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("TOPPADDING",    (0, 0), (0, 0),   30),
        ("TOPPADDING",    (0, 1), (0, 1),   6),
        ("BOTTOMPADDING", (0, 1), (0, 1),   30),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 14))

    # Status badge
    s_bg  = _PASS_BG if status == "VERIFIED" else _FAIL_BG
    s_fg  = _PASS_FG if status == "VERIFIED" else _FAIL_FG
    s_txt = "VERIFIED" if status == "VERIFIED" else "NOT VERIFIED"
    badge = Table(
        [[Paragraph(
            f"<b>Overall Verification Status:  {s_txt}</b>",
            ParagraphStyle("_badge", fontSize=13, fontName="Helvetica-Bold", textColor=s_fg)
        )]],
        colWidths=[16.5 * cm],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), s_bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(badge)
    story.append(Spacer(1, 14))

    # Layer score overview
    rows = [[
        Paragraph("Layer",        st["TblHdr"]),
        Paragraph("Rules Passed", st["TblHdr"]),
        Paragraph("Score",        st["TblHdr"]),
        Paragraph("Status",       st["TblHdr"]),
    ]]
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), _THEAD),
        ("GRID",       (0, 0), (-1, -1), 0.5, _DIVIDER),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]
    total_p, total_t = 0, 0
    for i, lk in enumerate(("correctness", "completeness", "consistency"), start=1):
        lr    = layers.get(lk, {})
        p, t, pct = _layer_score(lk, lr)
        total_p += p; total_t += t
        title, _, _ = LAYER_CATALOGUE[lk]
        lstatus = lr.get("status", "NOT RUN")
        bg = _PASS_BG if lstatus == "PASSED" else _FAIL_BG
        fg = _PASS_FG if lstatus == "PASSED" else _FAIL_FG
        rows.append([
            Paragraph(title, st["TblCell"]),
            Paragraph(f"{p} / {t}", st["TblCell"]),
            Paragraph(f"{pct}%", st["TblBold"]),
            Paragraph(
                f"<b>{lstatus}</b>",
                ParagraphStyle(f"_ls{i}", fontSize=9, fontName="Helvetica-Bold", textColor=fg)
            ),
        ])
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (2, i), _ROW_ALT))
        ts.append(("BACKGROUND", (3, i), (3, i), bg))

    ov_pct = int(total_p / total_t * 100) if total_t else 0
    rows.append([
        Paragraph("<b>Overall</b>", st["TblBold"]),
        Paragraph(f"<b>{total_p} / {total_t}</b>", st["TblBold"]),
        Paragraph(f"<b>{ov_pct}%</b>", st["TblBold"]),
        Paragraph("", st["TblCell"]),
    ])
    ts.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), _ACCENT))

    tbl = Table(rows, colWidths=[5 * cm, 4 * cm, 3.5 * cm, 4 * cm])
    tbl.setStyle(TableStyle(ts))
    story.append(tbl)


def _build_executive_summary(story, st, status, failed_layers, layers):
    story.append(Paragraph("<b>1. Executive Summary</b>", st["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_DIVIDER))
    story.append(Spacer(1, 8))

    total_p, total_t = 0, 0
    for lk in ("correctness", "completeness", "consistency"):
        p, t, _ = _layer_score(lk, layers.get(lk, {}))
        total_p += p; total_t += t
    ov_pct = int(total_p / total_t * 100) if total_t else 0

    if status == "VERIFIED":
        story.append(Paragraph(
            f"Verification Result: VERIFIED — {total_p}/{total_t} rules passed ({ov_pct}%)",
            st["StatusOK"]
        ))
        story.append(Paragraph(
            "The architectural description satisfies every rule across all three verification layers. "
            "It is structurally correct, architecturally complete, and internally consistent. "
            "The ADL is eligible to proceed to the validation phase.",
            st["Body"]
        ))
    else:
        story.append(Paragraph(
            f"Verification Result: NOT VERIFIED — {total_p}/{total_t} rules passed ({ov_pct}%)",
            st["StatusFail"]
        ))
        fl_str = ", ".join(f.capitalize() for f in failed_layers) if failed_layers else "—"
        story.append(Paragraph(
            f"Failed layers: <b>{fl_str}</b>. "
            "Violations and issues are detailed in the per-layer sections below. "
            "Apply the recommendations and re-run verification before proceeding.",
            st["Body"]
        ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Verification Scope</b>", st["H3"]))
    story.append(Paragraph(
        "Architecture verification examines the ADL description independently of quality attributes. "
        "It applies a fixed rule-set across three layers:",
        st["Body"]
    ))
    scope_rows = [
        [Paragraph("Layer", st["TblHdr"]),
         Paragraph("Focus", st["TblHdr"]),
         Paragraph("Rules", st["TblHdr"])],
        [Paragraph("Correctness",  st["TblCell"]),
         Paragraph("Structural and referential integrity — components and relationships are well-formed.", st["TblCell"]),
         Paragraph("7", st["TblBold"])],
        [Paragraph("Completeness", st["TblCell"]),
         Paragraph("Presence of required architectural elements — no isolation or gaps.", st["TblCell"]),
         Paragraph("3", st["TblBold"])],
        [Paragraph("Consistency",  st["TblCell"]),
         Paragraph("Internal coherence and style conformance — metrics and style agree.", st["TblCell"]),
         Paragraph("5", st["TblBold"])],
    ]
    scope_ts = _base_ts(3)
    scope_tbl = Table(scope_rows, colWidths=[3.5 * cm, 10 * cm, 3 * cm])
    scope_tbl.setStyle(TableStyle(scope_ts))
    story.append(scope_tbl)


def _build_layer_section(story, st, section_idx, layer_key, layer_result):
    title, rules, fk = LAYER_CATALOGUE[layer_key]
    status   = layer_result.get("status", "NOT RUN")
    findings = layer_result.get(fk, [])
    p, t, pct = _layer_score(layer_key, layer_result)

    story.append(Paragraph(f"<b>{section_idx}. {title} Layer Analysis</b>", st["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_DIVIDER))
    story.append(Spacer(1, 6))

    # Status + score badge
    s_bg = _PASS_BG if status == "PASSED" else _FAIL_BG
    s_fg = _PASS_FG if status == "PASSED" else _FAIL_FG
    badge = Table(
        [[Paragraph(
            f"<b>Layer Status: {status}</b>  |  Score: {p}/{t} rules passed ({pct}%)",
            ParagraphStyle(f"_lb_{layer_key}", fontSize=10, fontName="Helvetica-Bold", textColor=s_fg)
        )]],
        colWidths=[16.5 * cm],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), s_bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(badge)
    story.append(Spacer(1, 8))

    # Rule checklist
    story.append(Paragraph("<b>Rule Checklist</b>", st["H3"]))
    failed_set = {f.get("rule") for f in findings}
    rule_rows = [[
        Paragraph("Rule ID",     st["TblHdr"]),
        Paragraph("Description", st["TblHdr"]),
        Paragraph("Result",      st["TblHdr"]),
    ]]
    rts = [
        ("BACKGROUND", (0, 0), (-1, 0), _THEAD),
        ("GRID",       (0, 0), (-1, -1), 0.5, _DIVIDER),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ]
    for i, (rid, rdesc) in enumerate(rules, start=1):
        failed    = rid in failed_set
        r_bg      = _FAIL_BG if failed else _PASS_BG
        r_fg      = _FAIL_FG if failed else _PASS_FG
        result_lbl = "FAIL" if failed else "PASS"
        rule_rows.append([
            Paragraph(rid, st["TblCell"]),
            Paragraph(rdesc, st["TblCell"]),
            Paragraph(
                f"<b>{result_lbl}</b>",
                ParagraphStyle(f"_rr_{layer_key}_{i}", fontSize=9,
                               fontName="Helvetica-Bold", textColor=r_fg)
            ),
        ])
        if i % 2 == 0:
            rts.append(("BACKGROUND", (0, i), (1, i), _ROW_ALT))
        rts.append(("BACKGROUND", (2, i), (2, i), r_bg))

    rule_tbl = Table(rule_rows, colWidths=[4.5 * cm, 9.5 * cm, 2.5 * cm])
    rule_tbl.setStyle(TableStyle(rts))
    story.append(rule_tbl)
    story.append(Spacer(1, 10))

    # Observed findings
    if findings:
        story.append(Paragraph(f"<b>Observed Findings ({len(findings)})</b>", st["H3"]))
        find_rows = [[
            Paragraph("#",      st["TblHdr"]),
            Paragraph("Rule",   st["TblHdr"]),
            Paragraph("Detail", st["TblHdr"]),
        ]]
        fts = _base_ts(len(findings))
        for j, f in enumerate(findings, start=1):
            find_rows.append([
                Paragraph(str(j), st["TblCell"]),
                Paragraph(f.get("rule", "—"), st["TblBold"]),
                Paragraph(f.get("message", "—"), st["TblCell"]),
            ])
        find_tbl = Table(find_rows, colWidths=[1 * cm, 5 * cm, 10.5 * cm])
        find_tbl.setStyle(TableStyle(fts))
        story.append(find_tbl)
    else:
        story.append(Paragraph(
            "No findings — every rule in this layer passed.", st["Body"]
        ))
    story.append(Spacer(1, 14))


def _build_issues_table(story, st, layers):
    story.append(Paragraph("<b>6. Issues and Warnings</b>", st["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_DIVIDER))
    story.append(Spacer(1, 8))

    all_issues = []
    for lk in ("correctness", "completeness", "consistency"):
        _, _, fk = LAYER_CATALOGUE[lk]
        sev = _SEVERITY[lk]
        for f in layers.get(lk, {}).get(fk, []):
            all_issues.append((lk.capitalize(), sev, f.get("rule", "—"), f.get("message", "—")))

    if not all_issues:
        story.append(Paragraph(
            "No issues or warnings found. All verification layers passed cleanly.", st["Body"]
        ))
        return

    rows = [[
        Paragraph("Layer",    st["TblHdr"]),
        Paragraph("Severity", st["TblHdr"]),
        Paragraph("Rule",     st["TblHdr"]),
        Paragraph("Detail",   st["TblHdr"]),
    ]]
    ts = _base_ts(len(all_issues))
    for i, (layer, sev, rule, msg) in enumerate(all_issues, start=1):
        sev_bg = _FAIL_BG if sev == "High" else _WARN_BG
        sev_fg = _FAIL_FG if sev == "High" else _WARN_FG
        rows.append([
            Paragraph(layer, st["TblCell"]),
            Paragraph(
                f"<b>{sev}</b>",
                ParagraphStyle(f"_sv_{i}", fontSize=9, fontName="Helvetica-Bold", textColor=sev_fg)
            ),
            Paragraph(rule, st["TblBold"]),
            Paragraph(msg,  st["TblCell"]),
        ])
        ts.append(("BACKGROUND", (1, i), (1, i), sev_bg))

    tbl = Table(rows, colWidths=[3 * cm, 2.5 * cm, 4.5 * cm, 6.5 * cm])
    tbl.setStyle(TableStyle(ts))
    story.append(tbl)
    story.append(Spacer(1, 12))


def _build_recommendations(story, st, section_idx, verification_result):
    story.append(Paragraph(f"<b>{section_idx}. Recommendations</b>", st["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_DIVIDER))
    story.append(Spacer(1, 8))

    fixes = derive_suggested_fixes(verification_result)
    if fixes:
        for fix in fixes:
            story.append(Paragraph(f"&#8226; {fix}", st["Bullet"]))
    else:
        story.append(Paragraph(
            "No improvements required. The architecture satisfies every verification rule "
            "across all three layers.",
            st["Body"]
        ))
    story.append(Spacer(1, 12))


def _build_final_decision(story, st, section_idx, status, failed_layers):
    story.append(Paragraph(f"<b>{section_idx}. Final Verification Decision</b>", st["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_DIVIDER))
    story.append(Spacer(1, 8))

    if status == "VERIFIED":
        d_bg   = _PASS_BG
        d_fg   = _PASS_FG
        d_text = (
            "<b>DECISION: VERIFIED</b><br/><br/>"
            "The architectural description is correct, complete, and internally consistent. "
            "It satisfies all 15 verification rules across all three layers. "
            "The ADL is eligible to proceed to the architectural validation phase."
        )
    else:
        d_bg   = _FAIL_BG
        d_fg   = _FAIL_FG
        fl_str = ", ".join(f.capitalize() for f in failed_layers) if failed_layers else "—"
        d_text = (
            f"<b>DECISION: NOT VERIFIED</b><br/><br/>"
            f"Failed layers: {fl_str}.<br/><br/>"
            "The architectural description violates one or more verification rules. "
            "Until the violations listed in this report are resolved, the ADL is considered "
            "unfit for architectural validation. "
            "Apply the recommendations above and re-run verification."
        )

    decision_cell = Table(
        [[Paragraph(d_text,
                    ParagraphStyle("_decision", fontSize=11, fontName="Helvetica",
                                   textColor=d_fg, leading=17))]],
        colWidths=[16.5 * cm],
    )
    decision_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), d_bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(decision_cell)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_verification_pdf(verification_result: dict) -> str:
    status       = verification_result.get("status", "NOT_VERIFIED")
    layers       = verification_result.get("layers", {})
    failed_layers = verification_result.get("failed_layers", [])

    if status == "VERIFIED":
        path     = "data/outputs/architecture_verification_report.pdf"
        old_path = "data/outputs/architecture_verification_problems.pdf"
    else:
        path     = "data/outputs/architecture_verification_problems.pdf"
        old_path = "data/outputs/architecture_verification_report.pdf"

    os.makedirs("data/outputs", exist_ok=True)

    # Remove stale counterpart so the download endpoint always finds the right file
    if os.path.exists(old_path):
        os.remove(old_path)

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm,   bottomMargin=2.5 * cm,
    )
    st    = _styles()
    story = []

    # Cover page
    _build_cover(story, st, status, failed_layers, layers)
    story.append(PageBreak())

    # Executive summary
    _build_executive_summary(story, st, status, failed_layers, layers)
    story.append(PageBreak())

    # Per-layer analysis (sections 3-5)
    for idx, lk in enumerate(("correctness", "completeness", "consistency"), start=3):
        _build_layer_section(story, st, idx, lk, layers.get(lk, {}))

    story.append(PageBreak())

    # Issues & warnings (section 6)
    _build_issues_table(story, st, layers)

    # Recommendations (section 7)
    _build_recommendations(story, st, 7, verification_result)

    # Final decision (section 8)
    _build_final_decision(story, st, 8, status, failed_layers)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path