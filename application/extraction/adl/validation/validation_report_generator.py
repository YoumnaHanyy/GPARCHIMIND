from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# --------------------------------------------------
# Dynamic derivation of architectural fixes
# (Derived from validation metrics – NOT static)
# --------------------------------------------------
def derive_suggested_fixes(validation_result: dict):
    fixes = []

    structural = validation_result.get("structural", {})
    metrics = structural.get("metrics", {})

    if metrics.get("undefined_references", 0) > 0:
        fixes.append(
            "Undefined References: One or more relationships reference components "
            "that are not declared in the architectural model."
        )

    if metrics.get("self_loops", 0) > 0:
        fixes.append(
            "Self-Loops: One or more components have relationships where the "
            "source and target are the same component."
        )

    if metrics.get("invalid_relationship_types", 0) > 0:
        fixes.append(
            "Invalid Relationship Types: One or more relationships use interaction "
            "types outside the supported architectural vocabulary."
        )

    quality = validation_result.get("quality", {})
    centralization_index = quality.get("centralization_index", 0)

    if centralization_index > 0.6:
        fixes.append(
            "Architectural Centralization: A disproportionate number of interactions "
            "are concentrated on a limited subset of components."
        )

    for risk in quality.get("risks", []):
        fixes.append(
            f"High Coupling in Component '{risk.get('component')}': "
            "The component participates in a large number of dependencies."
        )

    domain = validation_result.get("domain", {})

    if domain.get("communication_style") == "synchronous":
        fixes.append(
            "Synchronous Communication Pattern: Components are dependent on "
            "immediate responses from other components at runtime."
        )

    if domain.get("state_concentration", 0) > 3:
        fixes.append(
            "State Concentration: System state handling is concentrated in a "
            "small number of components."
        )

    return fixes


# --------------------------------------------------
# PDF Validation Report Generator
# --------------------------------------------------
def generate_validation_pdf(validation_result: dict):

    status = validation_result["status"]

    if status == "FAILED":
        path = "data/outputs/architecture_validation_problems.pdf"
        title = "Architecture Validation Problems Report"
    else:
        path = "data/outputs/architecture_validation_report.pdf"
        title = "Architecture Validation & Quality Report"

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ==================================================
    # Title
    # ==================================================
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 14))

    # ==================================================
    # 1. Executive Summary
    # ==================================================
    story.append(Paragraph("<b>1. Executive Summary</b>", styles["Heading2"]))
    story.append(Paragraph(
        f"Overall Architecture Validation Result: <b>{status}</b>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    structural = validation_result["structural"]
    quality = validation_result["quality"]
    domain = validation_result["domain"]

    # ==================================================
    # 2. Structural Analysis
    # ==================================================
    story.append(Paragraph("<b>2. Structural Analysis</b>", styles["Heading2"]))

    metrics = structural["metrics"]
    story.append(Table(
        [["Metric", "Observed Value"]] +
        [[k.replace("_", " ").title(), v] for k, v in metrics.items()],
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black)
        ]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Metric Definitions and Calculation Rules</b><br/>"
        "Component Count: The total number of components explicitly declared "
        "in the architectural description.<br/>"
        "Relationship Count: The total number of relationships declared between components.<br/>"
        "Undefined References: The number of relationships whose source or target "
        "does not match any declared component name.<br/>"
        "Self-Loops: The number of relationships where the source and target "
        "components are identical.<br/>"
        "Invalid Relationship Types: The number of relationships whose interaction "
        "type is not recognized by the validation rules.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Structural Evaluation Interpretation</b><br/>"
        "Structural validation assesses the syntactic and referential correctness "
        "of the architecture. Any non-zero value in Undefined References, Self-Loops, "
        "or Invalid Relationship Types indicates an invalid architectural description.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # ==================================================
    # 3. Quality Analysis
    # ==================================================
    story.append(Paragraph("<b>3. Quality Analysis</b>", styles["Heading2"]))

    fan_in = quality.get("fan_in", {})
    fan_out = quality.get("fan_out", {})

    fan_table = [["Component", "Fan-In", "Fan-Out", "Coupling"]]
    for comp in fan_in:
        fan_table.append([
            comp,
            fan_in.get(comp, 0),
            fan_out.get(comp, 0),
            fan_in.get(comp, 0) + fan_out.get(comp, 0)
        ])

    story.append(Table(
        fan_table,
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black)
        ]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Calculation Formulae</b><br/>"
        "Fan-In(component) = Number of relationships where component is the target.<br/>"
        "Fan-Out(component) = Number of relationships where component is the source.<br/>"
        "Coupling(component) = Fan-In(component) + Fan-Out(component).",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Quality Evaluation Interpretation</b><br/>"
        "Components with high coupling are strongly dependent on other components. "
        "High coupling increases maintenance effort, change propagation, and the "
        "risk of cascading failures across the architecture.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # ==================================================
    # 4. Domain Analysis
    # ==================================================
    story.append(Paragraph("<b>4. Domain Analysis</b>", styles["Heading2"]))

    story.append(Table(
        [
            ["Metric", "Observed Value"],
            ["Stateless Ratio", domain.get("stateless_ratio")],
            ["State Concentration", domain.get("state_concentration")],
            ["Communication Style", domain.get("communication_style")]
        ],
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black)
        ]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Calculation Rules</b><br/>"
        "Stateless Ratio: Inferred value representing the proportion of components "
        "without explicitly declared state.<br/>"
        "State Concentration: Maximum Fan-In value observed across all components.<br/>"
        "Communication Style: Determined by comparing the number of event-driven "
        "relationships to synchronous data-flow relationships.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Domain Evaluation Interpretation</b><br/>"
        "High state concentration and synchronous communication patterns indicate "
        "centralized coordination and strong runtime dependencies, which negatively "
        "impact scalability, resilience, and fault tolerance.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    # ==================================================
    # 5. Suggested Improvements
    # ==================================================
    story.append(Paragraph("<b>5. Suggested Improvements</b>", styles["Heading2"]))
    for fix in derive_suggested_fixes(validation_result):
        story.append(Paragraph(f"- {fix}", styles["Normal"]))

    # ==================================================
    # 6. Final Validation Conclusion
    # ==================================================
    if status == "FAILED":
        story.append(Spacer(1, 14))
        story.append(Paragraph("<b>6. Final Validation Conclusion</b>", styles["Heading2"]))
        story.append(Paragraph(
            "The architecture failed validation due to accumulated structural, "
            "quality, and domain-level violations. Although the architecture may "
            "be executable, it does not conform to recommended architectural "
            "quality standards.",
            styles["Normal"]
        ))

    doc.build(story)
    return path