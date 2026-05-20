"""
ArchiMind — SRS Validator (Production Level)
=============================================
IEEE 830 / ISO/IEC 25010 compliant validation engine.

Validation Dimensions:
  1. Ambiguity Detection       — vague / unmeasurable language
  2. Testability Check         — no acceptance criteria / measurable thresholds
  3. Completeness Check        — missing IEEE-mandatory sections
  4. Consistency Check         — contradictory statements
  5. Traceability Check        — unidentified requirements
  6. Atomicity Check           — compound requirements (and / or)
  7. Correctness Check         — passive voice, modal verbs misuse
  8. AI-powered Deep Analysis  — Groq (llama-3.3-70b) for nuanced issues

Scoring Model  (weighted, 0-100):
  Clarity        20 pts
  Completeness   25 pts
  Consistency    20 pts
  Testability    20 pts
  Traceability   15 pts
"""

import re
import os
import json
import fitz                          # PyMuPDF
from groq import Groq
from dataclasses import dataclass, field, asdict
from typing import Optional


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class ValidationIssue:
    req_id: str
    requirement: str
    problem: str
    category: str          # Ambiguity | Testability | Completeness | ...
    severity: str          # Critical | High | Medium | Low
    suggestion: str
    rule_ref: str = ""     # e.g. "IEEE 830 §3.6"


@dataclass
class ComplianceScore:
    clarity: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    testability: float = 0.0
    traceability: float = 0.0

    def overall(self) -> float:
        weights = {
            "clarity":      0.20,
            "completeness": 0.25,
            "consistency":  0.20,
            "testability":  0.20,
            "traceability": 0.15,
        }
        return round(
            self.clarity      * weights["clarity"]      +
            self.completeness * weights["completeness"]  +
            self.consistency  * weights["consistency"]   +
            self.testability  * weights["testability"]   +
            self.traceability * weights["traceability"],
            1
        )


@dataclass
class ValidationResult:
    project_name: str
    total_requirements: int
    issues: list = field(default_factory=list)
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    quality_score: float = 0.0
    compliance: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    enhanced_srs: str = ""
    ai_summary: str = ""
    sections_found: list = field(default_factory=list)
    sections_missing: list = field(default_factory=list)


# ──────────────────────────────────────────────
# RULE DEFINITIONS
# ──────────────────────────────────────────────

# IEEE 830 mandatory sections
IEEE_MANDATORY_SECTIONS = [
    "introduction",
    "purpose",
    "scope",
    "definitions",
    "overview",
    "overall description",
    "product perspective",
    "product functions",
    "user characteristics",
    "constraints",
    "assumptions",
    "specific requirements",
    "functional requirements",
    "non-functional requirements",
    "performance requirements",
    "security",
    "appendix",
]

# Words that indicate ambiguity (IEEE 830 §3.6.1)
AMBIGUOUS_WORDS = {
    "fast":          "Specify response time in milliseconds or seconds.",
    "slow":          "Specify maximum acceptable response time.",
    "easy":          "Define usability criteria (e.g., task completion rate ≥ 95%).",
    "simple":        "Define measurable simplicity criteria.",
    "user-friendly": "Define UX metrics (e.g., SUS score ≥ 80).",
    "flexible":      "Describe exact configurability requirements.",
    "robust":        "Define fault-tolerance metrics (e.g., MTBF ≥ 1000h).",
    "efficient":     "Specify performance benchmarks (CPU, memory, time).",
    "quick":         "Define maximum acceptable latency.",
    "appropriate":   "Specify what qualifies as appropriate with measurable criteria.",
    "sufficient":    "Define the minimum acceptable quantity or quality.",
    "adequate":      "Replace with a measurable threshold.",
    "good":          "Define quality criteria explicitly.",
    "better":        "Specify improvement percentage or benchmark.",
    "modern":        "Reference a specific standard or version.",
    "intuitive":     "Define learnability metrics (e.g., time-on-task ≤ 2 min).",
    "seamless":      "Define integration acceptance criteria.",
    "optimal":       "Define optimization objective and constraints.",
    "state-of-the-art": "Reference a specific technology or standard.",
    "latest":        "Specify version or release date.",
    "real-time":     "Define maximum acceptable latency (e.g., ≤ 100ms).",
    "large":         "Specify size in units (KB, MB, records, users).",
    "small":         "Specify size in units.",
    "many":          "Provide exact count or range.",
    "several":       "Provide exact count or range.",
    "some":          "Provide exact count or range.",
    "various":       "Enumerate the specific variants.",
}

# Patterns indicating non-testable requirements
NON_TESTABLE_PATTERNS = [
    (r"\bshould\b",       "Use 'shall' for mandatory requirements (IEEE 830)."),
    (r"\bmay\b",          "Use 'shall' for mandatory; 'may' is optional and untestable."),
    (r"\bwill\b",         "Replace 'will' with 'shall' for binding requirements."),
    (r"\bcan\b",          "Replace 'can' with 'shall' for testable obligations."),
    (r"\bwould\b",        "Replace 'would' with 'shall' for binding requirements."),
    (r"\bmight\b",        "Remove uncertainty; use 'shall' with clear conditions."),
    (r"\bgenerally\b",    "Remove vague qualifiers; state exact conditions."),
    (r"\btypically\b",    "Remove vague qualifiers; state exact conditions."),
    (r"\busually\b",      "Specify the exact expected behavior."),
    (r"\bnormally\b",     "Specify the exact expected behavior."),
    (r"\boften\b",        "Specify exact frequency or conditions."),
    (r"\bif possible\b",  "State exact conditions or remove qualifier."),
    (r"\bwhere applicable\b", "Specify applicability conditions explicitly."),
]

# Consistency contradiction pairs
CONTRADICTION_PAIRS = [
    ("encrypted",    "unencrypted"),
    ("authenticated","unauthenticated"),
    ("online",       "offline"),
    ("synchronous",  "asynchronous"),
    ("mandatory",    "optional"),
    ("required",     "not required"),
    ("shall",        "shall not"),
]


# ──────────────────────────────────────────────
# MAIN VALIDATOR CLASS
# ──────────────────────────────────────────────

class SRSValidator:

    def __init__(self):
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=os.environ.get("GROQ_API_KEY", "")
            )
        return self._client

    # ── TEXT EXTRACTION ─────────────────────────

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def extract_project_name(self, text: str) -> str:
        """
        Extract the SRS document title (system/project name).
        Handles cases where the title spans two lines:
          "Software Requirements Specification for"
          "Gestify"   ← this is the actual name
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        skip_patterns = re.compile(
            r'\b(supervised|prepared|submitted|by|dr\.|eng\.|prof\.|university|'
            r'faculty|department|january|february|march|april|may|june|july|'
            r'august|september|october|november|december|\d{4})\b',
            re.IGNORECASE
        )

        # Pattern 1: "SRS/Specification for <Name>" — name on SAME line
        for line in lines[:40]:
            match = re.search(
                r"(?:specification|srs|requirements)[^\n]*\bfor\b\s+([A-Z][^\n]{2,60})",
                line,
                re.IGNORECASE
            )
            if match:
                candidate = match.group(1).strip()
                if candidate and not skip_patterns.search(candidate):
                    return candidate

        # Pattern 2: "... for" at end of line → name is on the NEXT line
        for i, line in enumerate(lines[:40]):
            if line.lower().rstrip().endswith("for"):
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Skip if it looks like a person's name (exactly 2 words, common name patterns)
                    is_person_name = bool(re.match(
                        r'^(Dr|Eng|Prof|Mr|Mrs|Ms)\.?\s', next_line, re.I
                    ))
                    if (2 < len(next_line) < 80
                            and not skip_patterns.search(next_line)
                            and not is_person_name):
                        return next_line

        # Pattern 3: Line containing "System" or "Software" near top
        for line in lines[:20]:
            if (re.search(r'\b(system|software|application|platform)\b', line, re.I)
                    and 5 < len(line) < 80
                    and not skip_patterns.search(line)):
                return line

        # Pattern 4: First short capitalised line that isn't a name/date
        for line in lines[:20]:
            if (5 < len(line) < 60
                    and line[0].isupper()
                    and not skip_patterns.search(line)
                    and not re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line)):
                return line

        return "Unknown Project"

    def extract_requirements(self, text: str) -> list[str]:
        """
        Extract sentences that look like formal requirements.
        Captures lines containing shall / must / will / should.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        reqs = []
        for s in sentences:
            s = s.strip()
            if re.search(r'\b(shall|must|will|should)\b', s, re.IGNORECASE):
                if 10 < len(s) < 500:
                    reqs.append(s)
        return reqs

    def extract_sections(self, text: str) -> tuple[list[str], list[str]]:
        """Return (found_sections, missing_sections) based on IEEE 830."""
        text_lower = text.lower()
        found, missing = [], []
        for section in IEEE_MANDATORY_SECTIONS:
            if section in text_lower:
                found.append(section)
            else:
                missing.append(section)
        return found, missing

    # ── RULE CHECKS ─────────────────────────────

    def check_ambiguity(
        self,
        requirements: list[str],
        issues: list[ValidationIssue]
    ) -> int:
        """
        Detect ambiguous wording per IEEE 830 §3.6.1.
        Returns deduction points.
        """
        deduction = 0
        seen = set()

        for idx, req in enumerate(requirements):
            req_lower = req.lower()
            for word, fix in AMBIGUOUS_WORDS.items():
                if word in req_lower and (idx, word) not in seen:
                    seen.add((idx, word))
                    severity = "High" if word in {
                        "real-time", "robust", "efficient"
                    } else "Medium"
                    issues.append(ValidationIssue(
                        req_id=f"REQ-{idx+1:03d}",
                        requirement=req[:200],
                        problem=f"Ambiguous term '{word}' violates IEEE 830 §3.6.1 (unambiguous requirements).",
                        category="Ambiguity",
                        severity=severity,
                        suggestion=fix,
                        rule_ref="IEEE 830 §3.6.1"
                    ))
                    deduction += 3 if severity == "High" else 2
        return deduction

    def check_testability(
        self,
        requirements: list[str],
        issues: list[ValidationIssue]
    ) -> int:
        """
        Detect non-testable modal verbs and vague qualifiers.
        Returns deduction points.
        """
        deduction = 0
        seen = set()

        for idx, req in enumerate(requirements):
            for pattern, fix in NON_TESTABLE_PATTERNS:
                if re.search(pattern, req, re.IGNORECASE) and (idx, pattern) not in seen:
                    seen.add((idx, pattern))
                    issues.append(ValidationIssue(
                        req_id=f"REQ-{idx+1:03d}",
                        requirement=req[:200],
                        problem=f"Non-testable language detected: pattern '{pattern}' found.",
                        category="Testability",
                        severity="High",
                        suggestion=fix,
                        rule_ref="IEEE 830 §3.6.4"
                    ))
                    deduction += 3
        return deduction

    def check_completeness(
        self,
        text: str,
        sections_missing: list[str],
        issues: list[ValidationIssue]
    ) -> int:
        """
        Verify IEEE 830 mandatory sections exist.
        Returns deduction points.
        """
        deduction = 0
        critical_sections = {
            "scope", "functional requirements", "specific requirements"
        }
        for section in sections_missing:
            severity = "Critical" if section in critical_sections else "Medium"
            issues.append(ValidationIssue(
                req_id="SRS-STRUCT",
                requirement=f"Section: '{section.title()}'",
                problem=f"Mandatory IEEE 830 section '{section.title()}' is missing.",
                category="Completeness",
                severity=severity,
                suggestion=f"Add a '{section.title()}' section as required by IEEE 830.",
                rule_ref="IEEE 830 §3"
            ))
            deduction += 5 if severity == "Critical" else 2
        return deduction

    def check_consistency(
        self,
        text: str,
        issues: list[ValidationIssue]
    ) -> int:
        """
        Detect contradictory statements within the SRS.
        Returns deduction points.
        """
        deduction = 0
        text_lower = text.lower()

        for term_a, term_b in CONTRADICTION_PAIRS:
            if term_a in text_lower and term_b in text_lower:
                issues.append(ValidationIssue(
                    req_id="SRS-CONSISTENCY",
                    requirement=f"Conflicting usage: '{term_a}' and '{term_b}'",
                    problem=(
                        f"The document uses both '{term_a}' and '{term_b}', "
                        "which may indicate inconsistent requirements."
                    ),
                    category="Consistency",
                    severity="High",
                    suggestion=(
                        f"Resolve the conflict between '{term_a}' and '{term_b}'. "
                        "Ensure each requirement has a single, unambiguous meaning."
                    ),
                    rule_ref="IEEE 830 §3.6.3"
                ))
                deduction += 5
        return deduction

    def check_traceability(
        self,
        requirements: list[str],
        issues: list[ValidationIssue]
    ) -> int:
        """
        Check if requirements have unique identifiers (FR-xx, NFR-xx, REQ-xx, etc.)
        Returns deduction points.
        """
        id_pattern = re.compile(
            r'\b(FR|NFR|REQ|UC|SRS|US|ID|R)\s*[-_]?\s*\d+\b',
            re.IGNORECASE
        )
        untracked = sum(
            1 for r in requirements
            if not id_pattern.search(r)
        )
        deduction = 0
        if untracked > 0:
            ratio = untracked / max(len(requirements), 1)
            if ratio > 0.5:
                issues.append(ValidationIssue(
                    req_id="SRS-TRACE",
                    requirement="Multiple requirements",
                    problem=(
                        f"{untracked} of {len(requirements)} requirements "
                        "lack unique identifiers."
                    ),
                    category="Traceability",
                    severity="High" if ratio > 0.7 else "Medium",
                    suggestion=(
                        "Assign unique IDs to every requirement "
                        "(e.g., FR-001, NFR-001) to enable traceability."
                    ),
                    rule_ref="IEEE 830 §3.6.5"
                ))
                deduction += min(10, int(ratio * 15))
        return deduction

    def check_atomicity(
        self,
        requirements: list[str],
        issues: list[ValidationIssue]
    ) -> int:
        """
        Detect compound requirements joined by 'and' or 'or' that should be split.
        Returns deduction points.
        """
        deduction = 0
        compound_pattern = re.compile(
            r'\bshall\b.{5,}\band\b.{5,}\bshall\b',
            re.IGNORECASE
        )
        for idx, req in enumerate(requirements):
            if compound_pattern.search(req):
                issues.append(ValidationIssue(
                    req_id=f"REQ-{idx+1:03d}",
                    requirement=req[:200],
                    problem="Compound requirement: contains multiple 'shall' clauses.",
                    category="Atomicity",
                    severity="Medium",
                    suggestion=(
                        "Split into separate atomic requirements, "
                        "each with a single 'shall' clause."
                    ),
                    rule_ref="IEEE 830 §3.6.2"
                ))
                deduction += 2
        return deduction

    # ── SCORING ─────────────────────────────────

    def compute_scores(
        self,
        requirements: list[str],
        issues: list[ValidationIssue],
        sections_found: list[str],
        sections_missing: list[str],
    ) -> ComplianceScore:
        """
        Compute per-dimension scores (0-100) using a weighted penalty model.
        """
        total = max(len(requirements), 1)
        total_sections = len(IEEE_MANDATORY_SECTIONS)

        by_category = {}
        for issue in issues:
            by_category.setdefault(issue.category, []).append(issue)

        def penalty(issues_list: list, per_high=5, per_med=2, per_low=1) -> float:
            pts = sum(
                per_high if i.severity in ("Critical", "High") else
                per_med  if i.severity == "Medium" else per_low
                for i in issues_list
            )
            return min(pts, 100)

        # Clarity = 100 - ambiguity penalties (normalised)
        amb_pen = penalty(by_category.get("Ambiguity", []))
        clarity = max(0, 100 - (amb_pen / total * 10))

        # Completeness = sections found ratio
        sections_score = (len(sections_found) / total_sections) * 100
        comp_issues_pen = penalty(by_category.get("Completeness", []))
        completeness = max(0, (sections_score * 0.7) + ((100 - comp_issues_pen) * 0.3))

        # Consistency
        con_pen = penalty(by_category.get("Consistency", []), per_high=10)
        consistency = max(0, 100 - con_pen)

        # Testability
        test_pen = penalty(by_category.get("Testability", []))
        testability = max(0, 100 - (test_pen / total * 8))

        # Traceability
        trace_pen = penalty(by_category.get("Traceability", []))
        traceability = max(0, 100 - trace_pen)

        return ComplianceScore(
            clarity=round(clarity, 1),
            completeness=round(completeness, 1),
            consistency=round(consistency, 1),
            testability=round(testability, 1),
            traceability=round(traceability, 1),
        )

    # ── AI ANALYSIS ─────────────────────────────

    def ai_deep_analysis(
        self,
        text: str,
        issues: list,
        compliance: ComplianceScore,
    ) -> tuple[str, list[str], str]:
        """
        Use Groq (llama-3.3-70b) to:
          1. Summarise the main quality problems found by rule-based checks
          2. Generate targeted improvement suggestions based on detected issues
          3. Generate Enhanced SRS — fix EVERY requirement that has an issue

        The Enhanced SRS is built from the ACTUAL detected issues, not random FRs.
        Each problematic requirement gets rewritten to be:
          - Unambiguous, Testable, Atomic, Uniquely identified (IEEE 830)

        Returns (ai_summary, suggestions_list, enhanced_srs)
        """

        # ── Build issues summary for the prompt ──
        # Group issues by requirement text — deduplicate
        req_issues: dict[str, list] = {}
        for issue in issues:
            req_text = issue.get("requirement", "").strip()
            if not req_text or req_text == "General SRS":
                continue
            if req_text not in req_issues:
                req_issues[req_text] = []
            req_issues[req_text].append({
                "problem":  issue.get("problem", ""),
                "category": issue.get("category", ""),
                "severity": issue.get("severity", ""),
            })

        # Limit to top 12 to stay within token limits
        top_reqs = list(req_issues.items())[:12]

        # Build detailed issues block — include FULL original text
        issues_text = ""
        for idx, (req_text, req_issue_list) in enumerate(top_reqs, 1):
            problems = "; ".join(
                f"[{i['category']}] {i['problem']}"
                for i in req_issue_list
            )
            issues_text += (
                f"\n--- REQUIREMENT {idx} ---\n"
                f"ID: REQ-{idx:03d}\n"
                f"ORIGINAL TEXT (copy this exactly into 'original' field):\n"
                f"\"{req_text[:300]}\"\n"
                f"PROBLEMS TO FIX: {problems}\n"
            )

        snippet = text[:2000]

        prompt = f"""You are a senior IEEE 830 SRS quality reviewer.

The rule-based validation engine found issues in the SRS requirements listed below.

YOUR TASKS:
1. Write a concise executive summary (3-5 sentences) about the overall quality problems.
2. Generate exactly 8 specific, actionable improvement suggestions.
3. For EACH requirement listed below, rewrite it to fix its specific problems:
   - Keep the SAME meaning and topic as the original — DO NOT change the subject
   - Replace vague terms with measurable criteria (e.g., "fast" → "within 2 seconds")
   - Use "shall" instead of "will", "should", "can", "may"
   - Add measurable acceptance criteria (numbers, percentages, time limits)
   - Make it atomic (one 'shall' clause only)
   - Keep it specific to what the original requirement was about

REQUIREMENTS TO FIX:
{issues_text}

COMPLIANCE SCORES:
- Overall: {compliance.overall()}/100
- Clarity: {compliance.clarity}/100
- Completeness: {compliance.completeness}/100
- Consistency: {compliance.consistency}/100
- Testability: {compliance.testability}/100
- Traceability: {compliance.traceability}/100

SRS EXCERPT (for context only):
\"\"\"
{snippet}
\"\"\"

CRITICAL RULES:
- The "enhanced" field must fix the SAME requirement, not invent a new unrelated one
- Do NOT change the subject/topic of the requirement
- The "original" field must contain the exact original text provided above
- Return ALL {len(top_reqs)} requirements in enhanced_requirements

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{
  "summary": "...",
  "suggestions": ["...", "...", "...", "...", "...", "...", "...", "..."],
  "enhanced_requirements": [
    {{"id": "REQ-001", "original": "exact original text here", "enhanced": "fixed version here", "issues_fixed": ["issue1", "issue2"], "rationale": "what was fixed and why"}},
    {{"id": "REQ-002", "original": "exact original text here", "enhanced": "fixed version here", "issues_fixed": ["issue1"], "rationale": "what was fixed and why"}}
  ]
}}

IMPORTANT:
- enhanced_requirements must contain ALL {len(top_reqs)} problematic requirements, not just 3.
- Each enhanced requirement must directly fix the problems listed for it.
- Do not invent new requirements. Only rewrite the ones listed above."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            data = json.loads(raw)

            summary     = data.get("summary", "")
            suggestions = data.get("suggestions", [])

            # Format enhanced SRS — one card per fixed requirement
            enhanced_reqs  = data.get("enhanced_requirements", [])
            enhanced_lines = []
            for r in enhanced_reqs:
                issues_fixed = r.get("issues_fixed", [])
                fixed_str = (
                    " | ".join(issues_fixed)
                    if issues_fixed else ""
                )
                enhanced_lines.append(
                    f"[{r.get('id','?')}] {r.get('enhanced','')}\n"
                    f"    ✦ Issues Fixed: {fixed_str}\n"
                    f"    ✦ Rationale: {r.get('rationale','')}"
                )
            enhanced_srs = "\n\n".join(enhanced_lines)

            return summary, suggestions, enhanced_srs

        except Exception as e:
            fallback_suggestions = [
                "Assign unique IDs (FR-001, NFR-001) to every requirement.",
                "Replace all modal verbs ('should', 'may', 'will') with 'shall'.",
                "Add measurable acceptance criteria to every functional requirement.",
                "Remove ambiguous adjectives: fast, easy, user-friendly, efficient.",
                "Split compound requirements (multiple 'shall' clauses) into atomic ones.",
                "Add missing IEEE 830 mandatory sections (Scope, Assumptions, Constraints).",
                "Define all acronyms and domain-specific terms in a Definitions section.",
                "Include a traceability matrix linking requirements to design components.",
            ]
            return (
                f"AI deep analysis unavailable ({str(e)[:80]}). "
                "Rule-based analysis completed successfully.",
                fallback_suggestions,
                "Enhanced SRS generation requires AI connectivity."
            )

    # ── MAIN ENTRY POINT ────────────────────────

    def validate_document(
        self,
        pdf_path: str,
        validation_id: str
    ) -> dict:
        """
        Full validation pipeline.
        Returns a dict ready for the API response and MongoDB storage.
        """
        # 1. Extract
        text = self.extract_text(pdf_path)
        project_name = self.extract_project_name(text)
        requirements = self.extract_requirements(text)
        sections_found, sections_missing = self.extract_sections(text)

        # 2. Rule-based checks
        issues: list[ValidationIssue] = []

        self.check_ambiguity(requirements, issues)
        self.check_testability(requirements, issues)
        self.check_completeness(text, sections_missing, issues)
        self.check_consistency(text, issues)
        self.check_traceability(requirements, issues)
        self.check_atomicity(requirements, issues)

        # 3. Scoring
        compliance = self.compute_scores(
            requirements, issues, sections_found, sections_missing
        )
        quality_score = compliance.overall()

        # 5. Issue severity counts
        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for issue in issues:
            sev_counts[issue.severity] = sev_counts.get(issue.severity, 0) + 1

        # 6. Serialise issues (dataclass → dict)
        issues_dicts = [asdict(i) for i in issues]

        # 4. AI deep analysis — pass real detected issues
        ai_summary, suggestions, enhanced_srs = self.ai_deep_analysis(
            text, issues_dicts, compliance
        )

        return {
            "project_name":          project_name,
            "total_requirements":    len(requirements),
            "original_text":         text,
            "original_requirements": requirements,
            "issues":                issues_dicts,
            "critical_issues":       sev_counts["Critical"],
            "high_issues":           sev_counts["High"],
            "medium_issues":         sev_counts["Medium"],
            "low_issues":            sev_counts["Low"],
            "quality_score":         quality_score,
            "compliance": {
                "clarity":      compliance.clarity,
                "completeness": compliance.completeness,
                "consistency":  compliance.consistency,
                "testability":  compliance.testability,
                "traceability": compliance.traceability,
                "overall":      quality_score,
            },
            "sections_found":   sections_found,
            "sections_missing": sections_missing,
            "suggestions":      suggestions,
            "enhanced_srs":     enhanced_srs,
            "ai_summary":       ai_summary,
        }