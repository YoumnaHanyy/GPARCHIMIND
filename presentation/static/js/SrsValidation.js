/**
 * ArchiMind — SRS Validation Studio  (JS v3)
 * ===========================================
 *
 * UX Flow:
 *   1.  User sees dashboard (project cards + "New Validation" card)
 *   2.  Clicks "New Validation" → #uploadView appears  (dashboard hidden)
 *   3.  User picks PDF and clicks "Start Validation"
 *   4.  ONLY #step-upload (the form) disappears — #uploadView stays
 *   5.  #loadingMessage appears with animated step progress
 *   6.  API call completes → loader hidden, #resultContent appears
 *   7.  User can navigate Phase 1→4 using Next / Previous buttons
 *
 * Phases:
 *   1 → Detected Issues
 *   2 → IEEE/ISO Compliance Analysis
 *   3 → Improvement Suggestions
 *   4 → Enhanced SRS
 */

"use strict";

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

let currentPhase  = 1;
let validationData = null;

const TOTAL_PHASES = 4;

const PHASE_META = {
  1: { name: "Detected Issues",      pct: 25,  tabLabel: "Detected Issues"      },
  2: { name: "Compliance Analysis",  pct: 50,  tabLabel: "Compliance Analysis"  },
  3: { name: "Improvement Suggestions", pct: 75, tabLabel: "Suggestions"        },
  4: { name: "Enhanced SRS",         pct: 100, tabLabel: "Enhanced SRS"         },
};

// ─────────────────────────────────────────────
// LOADER STEP ANIMATION
// ─────────────────────────────────────────────

const LOADER_STEPS = [
  { id: "lstep1", label: "Extracting requirements",    delay: 0    },
  { id: "lstep2", label: "Running ambiguity checks",   delay: 1500 },
  { id: "lstep3", label: "Checking IEEE 830 compliance", delay: 3500 },
  { id: "lstep4", label: "AI deep analysis",           delay: 6000 },
  { id: "lstep5", label: "Generating enhanced SRS",    delay: 10000 },
];

let _loaderTimers = [];

function startLoaderSteps() {
  _loaderTimers.forEach(clearTimeout);
  _loaderTimers = [];

  // Reset all steps
  LOADER_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) {
      el.className = "loader-step";
      el.innerHTML = `<i class="bi bi-hourglass-split me-2"></i>${s.label}`;
    }
  });

  // Mark first step active immediately
  _activateStep(0);

  // Schedule the rest
  LOADER_STEPS.forEach((s, idx) => {
    if (idx === 0) return;
    const t = setTimeout(() => _activateStep(idx), s.delay);
    _loaderTimers.push(t);
  });
}

function _activateStep(idx) {
  // Complete all prior steps
  for (let i = 0; i < idx; i++) {
    const el = document.getElementById(LOADER_STEPS[i].id);
    if (el) {
      el.className = "loader-step done";
      el.innerHTML = `<i class="bi bi-check2-circle me-2"></i>${LOADER_STEPS[i].label}`;
    }
  }
  // Activate current
  const el = document.getElementById(LOADER_STEPS[idx].id);
  if (el) {
    el.className = "loader-step active";
    el.innerHTML = `<i class="bi bi-arrow-right-circle me-2"></i>${LOADER_STEPS[idx].label}`;
  }
}

function stopLoaderSteps() {
  _loaderTimers.forEach(clearTimeout);
  _loaderTimers = [];
}

// ─────────────────────────────────────────────
// VIEW MANAGEMENT
// ─────────────────────────────────────────────

function showValidationUploader() {
  _hide("dashboardView");
  _show("uploadView");
  _show("step-upload");     // ensure form visible if we return here
  _hide("loadingMessage");
  _hide("resultContent");
  _hide("progressSection");
  _hide("validationSummary");
}

function backToValidationDashboard() {
  _show("dashboardView");
  _hide("uploadView");
  _hide("progressSection");
  stopLoaderSteps();
}

function backToDashboard() {
  // Go back to the project list (full page)
  window.location.href = "/srs-validations";
}

// ─────────────────────────────────────────────
// FORM SUBMIT — MAIN PIPELINE
// ─────────────────────────────────────────────

document.getElementById("processForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const file = document.getElementById("fileIn").files[0];

    if (!file) {
      _showErrorModal("Please select a PDF file before starting validation.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      _showErrorModal("Only PDF files are accepted.");
      return;
    }

    // ── Hide ONLY the upload form, keep uploadView visible ──
    _hide("step-upload");
    _hide("resultContent");
    _hide("validationSummary");
    _show("loadingMessage");
    startLoaderSteps();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/validate-srs", {
        method: "POST",
        body:   formData,
      });

      const result = await response.json();

      stopLoaderSteps();
      _hide("loadingMessage");

      if (!result.success) {
        _showErrorModal(result.message || "Validation failed. Please try again.");
        _show("step-upload");   // bring form back so user can retry
        return;
      }

      validationData = result;
      currentPhase   = 1;

      _populateSummary(result);
      _show("validationSummary");
      _populateAISummary(result);
      _show("progressSection");
      renderPhase();
      _show("resultContent");

    } catch (err) {
      console.error("Validation fetch error:", err);
      stopLoaderSteps();
      _hide("loadingMessage");
      _show("step-upload");
      _showErrorModal("A network error occurred. Please check your connection and try again.");
    }
  });

// ─────────────────────────────────────────────
// SUMMARY POPULATION
// ─────────────────────────────────────────────

function _populateSummary(result) {
  _setText("totalReqs",     result.total_requirements ?? 0);
  _setText("issuesFound",   result.issues?.length ?? 0);
  _setText("criticalIssues", result.critical_issues ?? 0);
  _setText("qualityScore",  `${result.quality_score ?? 0}%`);

  // Grade badge
  const badge = document.getElementById("gradeBadge");
  if (badge && result.grade) {
    badge.textContent = result.grade;
    badge.className   = `grade-badge grade-${result.grade.toLowerCase()} position-absolute`;
  }
}

function _populateAISummary(result) {
  const summary = result.ai_summary || "";
  if (summary && !summary.startsWith("AI deep analysis unavailable")) {
    _setText("aiSummaryText", summary);
    _show("aiSummaryBanner");
  }
}

// ─────────────────────────────────────────────
// PHASE RENDERING
// ─────────────────────────────────────────────

function renderPhase() {
  const meta = PHASE_META[currentPhase];

  // Progress bar
  document.getElementById("progressBar").style.width = meta.pct + "%";
  _setText("phaseLabel",   `Phase ${currentPhase}: ${meta.name}`);
  _setText("phasePercent", `${meta.pct}%`);

  // Tabs
  const tabContainer = document.getElementById("dynamicTabs");
  tabContainer.innerHTML = `
    <li class="nav-item">
      <a class="nav-link active" href="#" onclick="return false;">${meta.tabLabel}</a>
    </li>
  `;

  // Prev / Next buttons
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");

  btnPrev.style.visibility = currentPhase === 1 ? "hidden" : "visible";
  btnNext.innerHTML = currentPhase === TOTAL_PHASES
    ? '<i class="bi bi-check2 me-2"></i>Finish'
    : `Next Phase<i class="bi bi-arrow-right ms-2"></i>`;

  // Render tab content
  _renderTabContent(meta.tabLabel);
}

function _renderTabContent(tabLabel) {
  let html = "";

  if (!validationData) {
    html = `<p class="text-muted">No validation data available.</p>`;
  } else {
    switch (tabLabel) {
      case "Detected Issues":
        html = _renderDetectedIssues(validationData);
        break;
      case "Compliance Analysis":
        html = _renderCompliance(validationData);
        break;
      case "Suggestions":
        html = _renderSuggestions(validationData);
        break;
      case "Enhanced SRS":
        html = _renderEnhancedSRS(validationData);
        break;
      default:
        html = `<p class="text-muted">No data for this tab.</p>`;
    }
  }

  document.getElementById("tabContentBox").innerHTML = html;
}

// ─────────────────────────────────────────────
// PHASE NAVIGATION
// ─────────────────────────────────────────────

function changePhase(dir) {
  if (dir === 1 && currentPhase < TOTAL_PHASES) {
    currentPhase++;
    renderPhase();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else if (dir === -1 && currentPhase > 1) {
    currentPhase--;
    renderPhase();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else if (dir === 1 && currentPhase === TOTAL_PHASES) {
    // Finished — go back to dashboard and reload projects
    window.location.href = "/srs-validations";
  }
}

// ─────────────────────────────────────────────
// RENDER: DETECTED ISSUES  (Phase 1)
// ─────────────────────────────────────────────

function _renderDetectedIssues(data) {
  const issues = data.issues || [];

  if (issues.length === 0) {
    return `
      <div class="text-center py-5">
        <i class="bi bi-check-circle-fill text-success fs-1"></i>
        <h5 class="mt-3 fw-bold">No Validation Issues Found</h5>
        <p class="text-muted">Your SRS document looks clean!</p>
      </div>`;
  }

  // Group by category
  const grouped = {};
  issues.forEach(issue => {
    const cat = issue.category || "General";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(issue);
  });

  // Category icons
  const catIcon = {
    "Ambiguity":    "bi-question-circle",
    "Testability":  "bi-rulers",
    "Completeness": "bi-puzzle",
    "Consistency":  "bi-arrow-left-right",
    "Traceability": "bi-link-45deg",
    "Atomicity":    "bi-scissors",
    "General":      "bi-exclamation-circle",
  };

  let html = `<h5 class="section-header">Detected Validation Issues <span class="badge bg-danger ms-2">${issues.length}</span></h5>`;

  // Severity summary chips
  const sev = { Critical: 0, High: 0, Medium: 0, Low: 0 };
  issues.forEach(i => { sev[i.severity] = (sev[i.severity] || 0) + 1; });
  html += `<div class="d-flex gap-2 flex-wrap mb-4">`;
  if (sev.Critical) html += `<span class="issue-severity severity-critical">🔴 Critical: ${sev.Critical}</span>`;
  if (sev.High)     html += `<span class="issue-severity severity-high">🟠 High: ${sev.High}</span>`;
  if (sev.Medium)   html += `<span class="issue-severity severity-medium">🟡 Medium: ${sev.Medium}</span>`;
  if (sev.Low)      html += `<span class="issue-severity severity-low">🟢 Low: ${sev.Low}</span>`;
  html += `</div>`;

  // Issues by category
  for (const [category, catIssues] of Object.entries(grouped)) {
    const icon = catIcon[category] || "bi-exclamation-circle";
    html += `
      <div class="category-group mb-3">
        <div class="category-header">
          <i class="bi ${icon} me-2"></i>${category}
          <span class="badge bg-secondary ms-2">${catIssues.length}</span>
        </div>
    `;

    catIssues.forEach((issue, idx) => {
      const sevClass = {
        "Critical": "severity-critical",
        "High":     "severity-high",
        "Medium":   "severity-medium",
        "Low":      "severity-low",
      }[issue.severity] || "severity-low";

      const ruleRef = issue.rule_ref
        ? `<span class="rule-ref"><i class="bi bi-bookmark-check me-1"></i>${issue.rule_ref}</span>`
        : "";

      html += `
        <div class="validation-issue">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="issue-title">${issue.req_id || `REQ-${idx+1}`}</div>
            <div class="d-flex align-items-center gap-2">
              ${ruleRef}
              <div class="issue-severity ${sevClass}">${issue.severity}</div>
            </div>
          </div>
          <div class="req-desc mb-2">
            <strong>Requirement:</strong><br>
            <span class="req-text">${_escapeHtml(issue.requirement)}</span>
          </div>
          <div class="req-desc mb-2">
            <strong>Problem:</strong><br>${_escapeHtml(issue.problem)}
          </div>
          <div class="req-desc suggestion-block">
            <i class="bi bi-lightbulb text-warning me-1"></i>
            <strong>Suggestion:</strong> ${_escapeHtml(issue.suggestion)}
          </div>
        </div>`;
    });

    html += `</div>`;
  }

  return html;
}

// ─────────────────────────────────────────────
// RENDER: COMPLIANCE  (Phase 2)
// ─────────────────────────────────────────────

function _renderCompliance(data) {
  const c = data.compliance || {};

  const dimensions = [
    { key: "clarity",      label: "Clarity",      icon: "bi-eye",           desc: "Requirements are unambiguous and clearly expressed (IEEE 830 §3.6.1)." },
    { key: "completeness", label: "Completeness",  icon: "bi-puzzle",        desc: "All mandatory IEEE 830 sections and requirement types are present." },
    { key: "consistency",  label: "Consistency",   icon: "bi-arrow-repeat",  desc: "No contradictory statements; uniform terminology used throughout." },
    { key: "testability",  label: "Testability",   icon: "bi-rulers",        desc: "Requirements have measurable acceptance criteria." },
    { key: "traceability", label: "Traceability",  icon: "bi-link-45deg",    desc: "Each requirement has a unique identifier." },
  ];

  // Overall gauge
  const overall = c.overall ?? data.quality_score ?? 0;
  const gaugeColor = overall >= 80 ? "#10b981" : overall >= 60 ? "#f59e0b" : "#ef4444";

  let html = `
    <h5 class="section-header">IEEE 830 / ISO 25010 Compliance Analysis</h5>

    <div class="overall-gauge-wrapper text-center mb-5">
      <div class="overall-gauge" style="--gauge-color:${gaugeColor};--gauge-val:${overall};">
        <div class="gauge-inner">
          <span class="gauge-value">${overall.toFixed(0)}%</span>
          <span class="gauge-label">Overall</span>
        </div>
      </div>
      <p class="text-muted mt-2 small">Weighted compliance score across all IEEE 830 dimensions</p>
    </div>
  `;

  // Dimension bars
  dimensions.forEach(dim => {
    const val = c[dim.key] ?? 0;
    const barColor = val >= 80 ? "#10b981" : val >= 60 ? "#f59e0b" : "#ef4444";
    html += `
      <div class="compliance-dimension mb-4">
        <div class="d-flex align-items-center justify-content-between mb-1">
          <div class="d-flex align-items-center gap-2">
            <i class="bi ${dim.icon} text-secondary"></i>
            <span class="fw-bold">${dim.label}</span>
          </div>
          <span class="fw-bold" style="color:${barColor};">${val.toFixed(0)}%</span>
        </div>
        <p class="text-muted small mb-2" style="font-size:0.78rem;">${dim.desc}</p>
        <div class="compliance-bar">
          <div class="compliance-fill" style="width:${val}%;background:${barColor};"></div>
        </div>
      </div>`;
  });

  // Sections found / missing
  const found   = data.sections_found   || [];
  const missing = data.sections_missing || [];

  html += `
    <div class="row g-3 mt-3">
      <div class="col-md-6">
        <div class="sections-panel sections-found">
          <h6 class="fw-bold mb-3"><i class="bi bi-check-circle text-success me-2"></i>Sections Found (${found.length})</h6>
          ${found.map(s => `<div class="section-tag found">${_titleCase(s)}</div>`).join("")}
          ${found.length === 0 ? `<p class="text-muted small">None detected.</p>` : ""}
        </div>
      </div>
      <div class="col-md-6">
        <div class="sections-panel sections-missing">
          <h6 class="fw-bold mb-3"><i class="bi bi-x-circle text-danger me-2"></i>Sections Missing (${missing.length})</h6>
          ${missing.map(s => `<div class="section-tag missing">${_titleCase(s)}</div>`).join("")}
          ${missing.length === 0 ? `<p class="text-muted small">All sections present! ✓</p>` : ""}
        </div>
      </div>
    </div>`;

  return html;
}

// ─────────────────────────────────────────────
// RENDER: SUGGESTIONS  (Phase 3)
// ─────────────────────────────────────────────

function _renderSuggestions(data) {
  const suggestions = data.suggestions || [];

  if (suggestions.length === 0) {
    return `<p class="text-muted text-center py-4">No improvement suggestions generated.</p>`;
  }

  const icons = [
    "bi-1-circle", "bi-2-circle", "bi-3-circle", "bi-4-circle",
    "bi-5-circle", "bi-6-circle", "bi-7-circle", "bi-8-circle",
  ];

  let html = `
    <h5 class="section-header">
      Improvement Suggestions
      <span class="badge bg-primary ms-2">${suggestions.length}</span>
    </h5>
    <p class="text-muted small mb-4">
      Actionable recommendations to improve your SRS quality score, generated from
      IEEE 830 rule-based analysis and AI deep review.
    </p>`;

  suggestions.forEach((s, idx) => {
    const icon = icons[idx] || "bi-arrow-right-circle";
    html += `
      <div class="suggestion-card">
        <div class="suggestion-number">
          <i class="bi ${icon}"></i>
        </div>
        <div class="suggestion-text">
          ${_escapeHtml(s)}
        </div>
      </div>`;
  });

  return html;
}

// ─────────────────────────────────────────────
// RENDER: ENHANCED SRS  (Phase 4)
// ─────────────────────────────────────────────

function _renderEnhancedSRS(data) {
  const enhanced = data.enhanced_srs || "";

  if (!enhanced || enhanced.includes("requires AI connectivity")) {
    return `
      <div class="text-center py-4">
        <i class="bi bi-cloud-slash fs-1 text-muted"></i>
        <h5 class="mt-3">Enhanced SRS Unavailable</h5>
        <p class="text-muted">AI connectivity is required to generate enhanced requirements.</p>
      </div>`;
  }

  const blocks       = enhanced.split("\n\n").filter(b => b.trim());
  const validationId = data.validation_id || "";

  let html = `
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h5 class="section-header mb-0">
        <i class="bi bi-stars me-2 text-warning"></i>AI-Enhanced Requirements
        <span class="badge bg-success ms-2">${blocks.length} Fixed</span>
      </h5>
      ${validationId ? `
        <a href="/download-enhanced-srs/${validationId}"
           class="btn-download-pdf"
           target="_blank">
          <i class="bi bi-file-earmark-pdf me-2"></i>Download PDF
        </a>` : ""}
    </div>
    <p class="text-muted small mb-4">
      The following requirements were rewritten by AI to be unambiguous,
      testable, atomic, and uniquely identified per IEEE 830.
      Original issues detected are shown under each requirement.
    </p>`;

  blocks.forEach(block => {
    // Parse: [REQ-001] text \n ✦ Issues Fixed: ... \n ✦ Rationale: ...
    const issuesFixedMatch = block.match(/✦ Issues Fixed:\s*([^\n]*)/);
    const rationaleMatch   = block.match(/✦ Rationale:\s*([^\n]*)/);

    const issuesFixed = issuesFixedMatch ? issuesFixedMatch[1].trim() : "";
    const rationale   = rationaleMatch   ? rationaleMatch[1].trim()   : "";

    // Get the first line (ID + requirement text)
    const firstLine = block.split("\n")[0].trim();
    const idMatch   = firstLine.match(/^\[([^\]]+)\]/);
    const reqId     = idMatch ? idMatch[1] : "REQ";
    const reqBody   = idMatch ? firstLine.replace(/^\[[^\]]+\]\s*/, "") : firstLine;

    html += `
      <div class="enhanced-req-card mb-3">
        <div class="enhanced-req-header">
          <span class="req-id-badge">${_escapeHtml(reqId)}</span>
          <span class="enhanced-badge"><i class="bi bi-stars me-1"></i>Enhanced</span>
        </div>
        <div class="enhanced-req-body">${_escapeHtml(reqBody)}</div>
        ${issuesFixed ? `
          <div class="enhanced-issues-fixed">
            <i class="bi bi-check2-all text-success me-1"></i>
            <strong>Issues Fixed:</strong> ${_escapeHtml(issuesFixed)}
          </div>` : ""}
        ${rationale ? `
          <div class="enhanced-rationale">
            <i class="bi bi-lightbulb text-warning me-1"></i>
            <strong>Why:</strong> ${_escapeHtml(rationale)}
          </div>` : ""}
      </div>`;
  });

  return html;
}

// ─────────────────────────────────────────────
// OPEN EXISTING PROJECT
// ─────────────────────────────────────────────

async function openValidationProject(projectId) {
  showValidationUploader();

  _hide("step-upload");
  _show("loadingMessage");
  startLoaderSteps();

  try {
    const response = await fetch(`/validation-project/${projectId}`);
    const result   = await response.json();

    stopLoaderSteps();
    _hide("loadingMessage");

    if (!result.success) {
      _showErrorModal(result.message || "Failed to load validation project.");
      _show("step-upload");
      return;
    }

    validationData = result;
    currentPhase   = 1;

    _populateSummary(result);
    _show("validationSummary");
    _populateAISummary(result);
    _show("progressSection");
    renderPhase();
    _show("resultContent");

  } catch (err) {
    console.error("Open project error:", err);
    stopLoaderSteps();
    _hide("loadingMessage");
    _show("step-upload");
    _showErrorModal("Failed to load the validation project. Please try again.");
  }
}

// ─────────────────────────────────────────────
// SIDE MENU
// ─────────────────────────────────────────────

function toggleSideMenu() {
  document.getElementById("sideMenu").classList.toggle("open");
}

// ─────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────

function _show(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}

function _hide(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function _escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function _titleCase(str) {
  return str.replace(/\b\w/g, c => c.toUpperCase());
}

function _showErrorModal(message) {
  const msgEl = document.getElementById("errorModalMessage");
  if (msgEl) msgEl.textContent = message;
  const modal = new bootstrap.Modal(document.getElementById("errorModal"));
  modal.show();
}

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  console.log("ArchiMind SRS Validation Studio v3 — Ready");
});