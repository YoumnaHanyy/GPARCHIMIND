// Global state
let currentPhase = 1;
let extractedData = null;
let pendingConfirmation = false; 
// Add defensive check
window.addEventListener('DOMContentLoaded', () => {
  console.log("Dashboard initialized");
  if (extractedData) {
    console.log("Project ID:", extractedData.project_id);
  }
});

  const phaseData = {
    1: { 
      name: "Requirement Analysis", 
      pct: "25%", 
      tabs: ["Functional Requirements", "Non-Functional Requirements"] 
    },
    2: { 
      name: "AI Recommendation", 
      pct: "66%", 
      tabs: [
        "Suggested Architecture",
        "Functional Method",
        "Ordinal Method",
        "Binary Method",
        "Weighted Method"
      ] 
    },
    3: { 
      name: "Final report", 
      pct: "90%", 
      tabs: ["Final Report"] 
    },
     4: { 
    name: "Design Patterns & Code", 
    pct: "100%", 
    tabs: ["Design Patterns", "Code Skeleton"] 
  }
  };

  function showUploader() {
    document.getElementById('dashboardView').classList.add('hidden');
    document.getElementById('uploadView').classList.remove('hidden');
  }

  /* =======================
     LOADER SELECTION (ONE RANDOM LOADER)
  ======================= */
  const loaders = ['layeredLoader', 'clientServerLoader', 'eventDrivenLoader'];

  function showRandomLoader() {
    // Hide all loaders
    loaders.forEach(id => {
      document.getElementById(id).classList.add('hidden');
    });

    // Pick one random loader
    const randomIndex = Math.floor(Math.random() * loaders.length);
    const selectedLoader = loaders[randomIndex];
    
    // Show selected loader
    document.getElementById(selectedLoader).classList.remove('hidden');

    // Handle event-driven animation
    if (selectedLoader === 'eventDrivenLoader') {
      startEventDrivenAnimation();
    }
  }

  function startLoadingAnimation() {
    showRandomLoader();
  }

  function stopLoadingAnimation() {
    // Hide all loaders
    loaders.forEach(id => {
      document.getElementById(id).classList.add('hidden');
    });
  }

  /* =======================
     EVENT-DRIVEN ANIMATION
  ======================= */
  function startEventDrivenAnimation() {
    const bell = document.getElementById("bell");
    const consumer = document.getElementById("consumer");
    const zzz = document.getElementById("zzz");
    const title = document.getElementById("edTitle");
    const desc = document.getElementById("edDesc");

    function sleepState() {
      bell.style.display = "none";
      bell.style.animation = "none";
      consumer.classList.add("sleeping");
      consumer.style.animation = "none";
      zzz.style.display = "block";
      title.innerText = "Processing your document… please wait";
      desc.innerText = "In Event-Driven Architecture, the system waits for events and reacts only when an event is triggered";
      setTimeout(eventArrives, 2000);
    }

    function eventArrives() {
      bell.style.display = "block";
      bell.style.animation = "bellRing 0.6s infinite";
      setTimeout(() => {
        zzz.style.display = "none";
        consumer.classList.remove("sleeping");
        consumer.style.animation = "work 0.3s infinite";
        title.innerText = "Processing your document… please wait";
        desc.innerText = "In Event-Driven Architecture, the system waits for events and reacts only when an event is triggered";
      }, 1000);
      setTimeout(sleepState, 3000);
    }

    sleepState();
  }

  /* =======================
     RENDER FUNCTIONS
  ======================= */

  function renderFunctionalRequirements(data) {
    if (!data || !data.functional || data.functional.length === 0) {
      return "<p class='text-muted'>No functional requirements found.</p>";
    }

    let html = "<h5 class='section-header'>Functional Requirements</h5>";

    data.functional.forEach((fr, idx) => {
      html += `
        <div class="mb-3">
          <div class="req-desc"><strong>FR-${String(idx + 1).padStart(2, '0')}:</strong>
          ${fr.description || 'No description'}</div>
        </div>
      `;
    });

    return html;
  }

  function renderNonFunctionalRequirements(data) {
    if (!data || !data.nfr_predictions || data.nfr_predictions.length === 0) {
      return "<p class='text-muted'>No non-functional requirements found.</p>";
    }

    let html = "<h5 class='section-header'>Non-Functional Requirements</h5>";

    data.nfr_predictions.forEach((nfr, idx) => {
      html += `
        <div class="mb-3">
          <div class="req-title">${String.fromCharCode(97 + idx)}. ${nfr.predicted_type_label || 'Unknown Type'}</div>
          <div class="req-desc">${nfr.description || 'No description'}</div>
        </div>
      `;
    });

    return html;
  }

  function renderFunctionalMethod(data) {
    if (!data || !data.functional_method || !data.functional_method.top_architectures) {
      return "<p class='text-muted'>No functional method results.</p>";
    }

    let html = "<h5 class='section-header'>Functional Architecture Method</h5>";

    data.functional_method.top_architectures.forEach((item, idx) => {
      html += `
        <div class="mb-4">
          <div class="req-title">
            ${idx + 1}. ${item.Architecture.replace("_", " ").toUpperCase()}
          </div>
          <div class="req-desc"><strong>Score:</strong> ${item.Score}</div>
          <div class="req-desc">${item.Reason || 'No reason provided'}</div>
        </div>
      `;
    });

    return html;
  }

  function renderOrdinalMethod(data) {
    if (!data || !data.ordinal_method || data.ordinal_method.length === 0) {
      return "<p class='text-muted'>No ordinal method results available.</p>";
    }

    let html = "<h5 class='section-header'>Ordinal Method</h5>";

    data.ordinal_method.forEach((item, idx) => {
      html += `
        <div class="mb-3">
          <div class="req-title">${idx + 1}. ${item.architecture}</div>
          <div class="req-desc">
            Matched NFRs: <strong>${item.matched_nfrs}</strong>
          </div>
        </div>
      `;
    });

    return html;
  }

  function renderBinaryMethod(data) {
    if (!data || !data.binary_method || !data.binary_method.top_architectures) {
      return "<p class='text-muted'>No binary method results available.</p>";
    }

    let html = "<h5 class='section-header'>Binary Method</h5>";

    data.binary_method.top_architectures.forEach((item, idx) => {
      html += `
        <div class="mb-3">
          <div class="req-title">${idx + 1}. ${item.architecture}</div>
          <div class="req-desc">
            Score: <strong>${item.score}</strong>
          </div>
        </div>
      `;
    });

    return html;
  }

  function renderWeightedMethod(data) {
    if (!data || !data.weighted_method || !data.weighted_method.top_architectures) {
      return "<p class='text-muted'>No weighted method results available.</p>";
    }

    let html = "<h5 class='section-header'>Weighted Score Method</h5>";

    data.weighted_method.top_architectures.forEach((item, idx) => {
      html += `
        <div class="mb-3">
          <div class="req-title">${idx + 1}. ${item.Architecture}</div>
          <div class="req-desc">
            Score: <strong>${item.Score}</strong>
          </div>
        </div>
      `;
    });

    return html;
  }

  function renderHybridMethod(data) {
    if (!data || !data.hybrid_method || data.hybrid_method.length === 0) {
      return "<p class='text-muted'>No Final Recommendation Available.</p>";
    }

    let html = "<h5 class='section-header'>Top Recommended Architectures</h5>";
    html += "<p class='text-muted mb-4'>Aggregated from Functional, Ordinal, Binary, and Weighted methods.</p>";

    data.hybrid_method.forEach((item, idx) => {
      const badge = idx === 0 ? '<span class="badge bg-success ms-2">Recommended</span>' : '';
      html += `
        <div class="mb-4">
          <div class="req-title">
            ${idx + 1}. ${item.Architecture.replace("_", " ").toUpperCase()} ${badge}
          </div>
          <div class="req-desc">
            <strong>Final Score:</strong> ${item.FinalScore}%
          </div>
        </div>
      `;
    });

    return html;
  }

  function renderFinalReport(data) {
    if (!data || !data.hybrid_method || data.hybrid_method.length === 0) {
   return "<p class='text-muted'>No data available for final report.</p>";
    }

    const topArch = data.hybrid_method[0];
    
    let html = "<h5 class='section-header'>Final Architecture Recommendation</h5>";
    html += `
     <div class="alert alert-success border-0 shadow-sm">
        <h4 class="alert-heading">
          <i class="bi bi-check-circle-fill me-2"></i>
          Recommended Architecture: ${topArch.Architecture.replace("_", " ").toUpperCase()}
        </h4>
        <p class="mb-0">Final Score: <strong>${topArch.FinalScore}%</strong></p>
      </div>
      <h6 class="mt-4 mb-3 text-secondary">Top 5 Architecture Recommendations (Select your preference):</h6>
    `;

    data.hybrid_method.forEach((item, idx) => {
       html += `
        <div class="arch-item d-flex justify-content-between align-items-center mb-2 p-3" 
             onclick="selectArchitecture(this, '${item.Architecture}')">
          <span class="fw-medium">${idx + 1}. ${item.Architecture.replace("_", " ").toUpperCase()}</span>
          <span class="badge rounded-pill bg-primary">${item.FinalScore}%</span>
        </div>`;
    });

    html += `
      <div class="mt-4 p-3 bg-light rounded-3 d-flex justify-content-between align-items-center">
        <div>
            <button id="saveArchBtn" class="btn btn-primary rounded-pill px-4" onclick="saveSelectedArchitecture()">
                <i class="bi bi-cloud-arrow-up-fill me-2"></i>Confirm & Save Choice
            </button>
        </div>
        <div>
            <button id="generateBtn" class="btn btn-success rounded-pill px-4 btn-disabled-locked" onclick="generateADL()" disabled>
                <i class="bi bi-file-earmark-pdf-fill me-2"></i>Generate ADL Blueprint
                </button>
            <button id="viewBlueprintBtn" class="btn btn-sm rounded-pill px-3 d-none" onclick="openPreviewModal()">
      <i class="bi bi-eye-fill me-1"></i> View Your ADL Blueprint
    </button>
            </button>
        </div>
      </div>`;

    return html;
  }


let selectedArchitecture = null;

function selectArchitecture(el, architecture) {
  // remove selection from all
  document.querySelectorAll('.arch-item').forEach(item => {
    item.classList.remove('selected-arch');
  });

  // mark selected
  el.classList.add('selected-arch');

  selectedArchitecture = architecture;
  console.log("Selected Architecture:", selectedArchitecture);
}


async function saveSelectedArchitecture() {
   if (!selectedArchitecture) {
        alert("Please select an architecture from the list first! 👇");
        return;
    }

    const saveBtn = document.getElementById('saveArchBtn');
    const genBtn = document.getElementById('generateBtn');

    try {
        saveBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span> Saving...`;
        saveBtn.disabled = true;

        const res = await fetch("/api/save-hybrid-result", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_id: extractedData.project_id,
                selected_architecture: selectedArchitecture,
                hybrid_result: extractedData.hybrid_method
            })
        });

        if (!res.ok) throw new Error("Save failed");

        // --- SUCCESS STATE ---
        saveBtn.classList.replace('btn-primary', 'btn-success');
        saveBtn.innerHTML = `<i class="bi bi-check-lg me-2"></i>Choice Saved!`;
        
        // UNLOCK Generate Button
        genBtn.disabled = false;
        genBtn.classList.remove('btn-disabled-locked');
        genBtn.classList.add('animate-bounce'); // Optional: add a little bounce to get attention

    } catch (err) {
        console.error(err);
        saveBtn.classList.replace('btn-primary', 'btn-danger');
        saveBtn.innerHTML = `<i class="bi bi-x-circle me-2"></i>Error. Try Again?`;
        saveBtn.disabled = false;
    }
}

async function loadPhase4() {
  const res = await fetch(`/phase4/${extractedData.project_id}`);
  const data = await res.json();

  console.log("PHASE4 RESPONSE:", data);

  extractedData.phase4 = data.phase4;

  renderPhase();
}

function renderDesignPatterns(data) {
  if (
    !data ||
    !data.phase4 ||
    !Array.isArray(data.phase4.top_patterns) ||
    data.phase4.top_patterns.length === 0
  ) {
     return "<p class='text-muted'>Loading design patterns...</p>";
  }
  let html = "<h5 class='section-header'>Recommended Design Patterns</h5>";

  data.phase4.top_patterns.forEach((p, idx) => {
    html += `
    
      <div class="mb-4">
        <div class="req-title">${idx + 1}. ${p.pattern}</div>
        <div class="req-desc">
          ${Array.isArray(p.reasons) ? p.reasons.join(", ") : "No reasons available"}
        </div>
      </div>
    `;
  });

  return html;
}

function renderCodeSkeleton(data) {
  if (!data || !data.phase4 || !data.phase4.code) {
    return "<p class='text-muted'>No code generated.</p>";
  }

  return `
    <h5 class="section-header">Generated Code Skeleton</h5>

    <div class="code-box position-relative">
      <button class="copy-btn" onclick="copyCode()">Copy</button>
      <pre><code id="generatedCode">${data.phase4.code}</code></pre>
    </div>

    <div class="mt-4 text-center">
      <button class="btn btn-success px-4 me-2" onclick="downloadCode()">Download</button>
      <button class="btn btn-outline-light px-4" onclick="regenerateCode()">Regenerate</button>
    </div>
  `;
}
  /* =======================
     TAB RENDERING
  ======================= */
  let currentPdfUrl = null;
async function generateADL() {
   console.log("Generate ADL clicked");
console.log("Project ID:", extractedData?.project_id);

    // 1. Show Loading Animation
    document.getElementById('resultContent').classList.add('hidden');
    const loading = document.getElementById('loadingMessage');
    loading.classList.remove('hidden');
    startLoadingAnimation();

    try {
        if (!extractedData?.project_id) {
            alert("Project ID not found. Please extract first.");
            return;
        }

        const response = await fetch(`/generate/${extractedData.project_id}`);
        
          if (!response.ok) throw new Error("Server response was not ok");

        // 3. Ne7awel el response le "Blob" (Binary Large Object)
        const blob = await response.blob();
        
        // 4. Ne3mel link "fake" 3ashan n-trigger el download
           // 3. Create a clean URL for the PDF blob
        if (currentPdfUrl) {
        URL.revokeObjectURL(currentPdfUrl);
}

        currentPdfUrl = window.URL.createObjectURL(blob);
        openPreviewModal();

        // --- EL GDEED: Show "View" button w hide "Generate" aw khallihom ganb ba3d ---
        const viewBtn = document.getElementById('viewBlueprintBtn');
        if(viewBtn) viewBtn.classList.remove('d-none');
   
        // 4. Update el Iframe SRC
        const frame = document.getElementById('reportFrame');
        frame.src = currentPdfUrl;

        // 5. Open el Modal (Make sure Bootstrap is loaded)
         const reportModal = new bootstrap.Modal(
            document.getElementById('reportModal')
        );
        reportModal.show();
        document.getElementById('reportModal').addEventListener('hidden.bs.modal', () => {
        document.body.classList.remove('modal-open');

        document.querySelectorAll('.modal-backdrop').forEach(el => el.remove());
        });
        // 6. Cleanup el memory lma el modal ye2fel
         document
          .getElementById('reportModal')
          .addEventListener(
              'hidden.bs.modal',
              () => window.URL.revokeObjectURL(currentPdfUrl),
              { once: true }
          );

    } catch (err) {
         } finally {
        // 7. Hide Loading
        // 6. El Loading haye2f hna awel ma el sater bta3 el 'await' ykhallas
        stopLoadingAnimation();
        loading.classList.add('hidden');
        document.getElementById('resultContent').classList.remove('hidden');
    }
}
const frame = document.getElementById('reportFrame');
const loader = document.getElementById('modalIframeLoader');
if (frame) {
frame.onload = function() {
        loader.style.display = 'none'; // Ekhfy el spinner
        frame.style.opacity = '1';    // Fade in el PDF
    };
}

function openPreviewModal() {
    if (!currentPdfUrl) {
        alert("No blueprint generated yet!");
        return;
    }
    
    const frame = document.getElementById('reportFrame');
    frame.src = currentPdfUrl;
    frame.style.opacity = '1';
    
    const reportModal = new bootstrap.Modal(document.getElementById('reportModal'));
    
}


function loadValidationReport() {
  const frame = document.getElementById("reportFrame");
  const loader = document.getElementById("modalIframeLoader");

  loader.style.display = "block";
  frame.style.opacity = "0";

  frame.src = "/download-validation-report";

  frame.addEventListener(
    "load",
    () => {
      loader.style.display = "none";
      frame.style.opacity = "1";
    },
    { once: true }
  );
}


  function renderPhase() {
    const data = phaseData[currentPhase];
    
    // Update Progress
    document.getElementById('progressBar').style.width = data.pct;
    document.getElementById('phaseLabel').innerText = `Phase ${currentPhase}: ${data.name}`;
    document.getElementById('phasePercent').innerText = data.pct;

    // Update Tabs
    const tabContainer = document.getElementById('dynamicTabs');
    tabContainer.innerHTML = data.tabs.map((t, i) => `
      <li class="nav-item">
        <a class="nav-link ${i===0?'active':''}" href="#" onclick="setTab(this, '${t}')">${t}</a>
      </li>
    `).join('');

    // Update Buttons
    document.getElementById('btnPrev').style.visibility = currentPhase === 1 ? 'hidden' : 'visible';
    document.getElementById('btnNext').innerText = currentPhase === 4 ? "Finish" : "Next Phase";

    // Load initial content
    setTab(null, data.tabs[0]);
  }

  function setTab(el, name) {
    if(el) {
        document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
        el.classList.add('active');
    }
    
    let content = '';
    
    if (!extractedData) {
      content = "<p class='text-muted'>No data available.</p>";
    } else {
      switch(name) {
        case "Functional Requirements":
          content = renderFunctionalRequirements(extractedData);
          break;
        case "Non-Functional Requirements":
          content = renderNonFunctionalRequirements(extractedData);
          break;
        case "Functional Method":
          content = renderFunctionalMethod(extractedData);
          break;
        case "Ordinal Method":
          content = renderOrdinalMethod(extractedData);
          break;
        case "Binary Method":
          content = renderBinaryMethod(extractedData);
          break;
        case "Weighted Method":
          content = renderWeightedMethod(extractedData);
          break;
        case "Suggested Architecture":
          content = renderHybridMethod(extractedData);
          break;
        case "Final Report":
          content = renderFinalReport(extractedData);
          break;
        case "Design Patterns":
          content = renderDesignPatterns(extractedData);
          break;
        case "Code Skeleton":
          content = renderCodeSkeleton(extractedData);
          break;
        default:
          content = `<p class='text-muted'>Rendering ${name} data...</p>`;
      }
    }
    
    document.getElementById('tabContentBox').innerHTML = content;
  }

function changePhase(dir) {

  if (pendingConfirmation) {
    new bootstrap.Modal(
      document.getElementById("warningModal")
    ).show();
    return;
  }
    

    if (dir === 1 && currentPhase < 4) {
    currentPhase++;

    if (currentPhase === 4) {
       loadPhase4(); // 🔥 يستنى الداتا الأول
    }

    renderPhase(); // بعد ما الداتا وصلت

    syncProjectProgress();
    triggerLoading();
  } else if (dir === -1 && currentPhase > 1) {
    currentPhase--;
    renderPhase();

  } else if (dir === 1 && currentPhase === 4) {
    syncProjectProgress();
    alert("Project Complete! Returning to dashboard.");
    location.reload();
  }
  

}



  function triggerLoading() {
    document.getElementById('resultContent').classList.add('hidden');
    document.getElementById('loadingMessage').classList.remove('hidden');
    startLoadingAnimation();
    
    setTimeout(() => {
        stopLoadingAnimation();
        document.getElementById('loadingMessage').classList.add('hidden');
        document.getElementById('resultContent').classList.remove('hidden');
        renderPhase();
    }, 5000);
  }

  /* =======================
     FORM SUBMISSION
  ======================= */

// ✅ UPDATED: Process Form Submission
document.getElementById('processForm').onsubmit = async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('fileIn');
  const loading = document.getElementById('loadingMessage');
  const errorBox = document.getElementById('errorMessage');
  
  if (!fileInput.files.length) {
    showErrorModal("Please select a PDF file before continuing.");
    return;
  }

  // Hide upload step, show loading
  document.getElementById('step-upload').classList.add('hidden');
  loading.classList.remove('hidden');
  errorBox.innerHTML = "";
  startLoadingAnimation();

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/extract", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) throw data;

    // Store extracted data
    extractedData = data;
    window.currentProjectId = data.project_id;
    
    if (data.srs_verified) {
    showSrsVerifiedBadge();
   }

    // Stop loading animation
    stopLoadingAnimation();

    // ✅ CHECK if confirmation needed
    if (data.needs_confirmation && data.low_confidence_nfrs.length > 0) {
      // Show modal, DON'T show results yet
      pendingConfirmation = true;
      loading.classList.add('hidden');
      openNFRConfirmModal(data.low_confidence_nfrs);
    } else {
      // No confirmation needed, show results directly
      showResults();
    }


  } catch (err) {
      console.log("🔥 ERROR FROM BACKEND:", err);

    stopLoadingAnimation();
    loading.classList.add('hidden');
    document.getElementById('step-upload').classList.remove('hidden');

    let msg = err.error || err.message || "Unknown error occurred";
    let friendlyMsg = "Unexpected error occurred. Please try again.";

    if (msg.toLowerCase().includes("pdf")) {
      friendlyMsg = "Invalid file format. Please upload a valid PDF document.";
    } else if (msg.toLowerCase().includes("timeout")) {
      friendlyMsg = "The process took too long. Please try again later.";                            
    }

    showErrorModal(friendlyMsg);
  }
};

// ✅ NEW: Show results helper
function showResults() {
  document.getElementById('loadingMessage').classList.add('hidden');
  document.getElementById('progressSection').classList.remove('hidden');
  document.getElementById('resultContent').classList.remove('hidden');
  renderPhase();
}

// ✅ UPDATED: NFR Confirmation Modal
function openNFRConfirmModal(nfrs) {
  if (!extractedData || !extractedData.project_id) {
    console.error("Cannot open NFR modal: project_id not available");
    showErrorModal("Project ID not found. Please try extracting again.");
    return;
  }

  const body = document.getElementById("nfrModalBody");
  body.innerHTML = "";

  nfrs.forEach((nfr, idx) => {
    body.innerHTML += `
  <div class="card shadow-sm mb-4 border-0">
    <div class="card-body">

      <div class="d-flex justify-content-between align-items-start mb-2">
        <h6 class="fw-bold mb-0">NFR (Low confidence)</h6>
        <span class="badge bg-danger">
          ${(nfr.confidence * 100).toFixed(0)}%
        </span>
      </div>

      <p class="fw-semibold mb-1">
        Title: ${nfr.title || "N/A"}
      </p>

      <p class="text-muted small mb-2">
        ${nfr.description}
      </p>

      <p class="small mb-3">
        <strong>Model guess:</strong>
        <span class="text-primary">
          ${nfr.predicted_type_label || "Unknown"}
        </span>
        (${(nfr.confidence * 100).toFixed(0)}%)
      </p>

      <div class="row g-2 align-items-center">
        <div class="col-12">
          <select class="form-select nfr-user-choice"
                  data-desc="${nfr.description}">
            <option value="">-- Select NFR Type --</option>
            <option value="A">Availability</option>
            <option value="FT">Fault Tolerance</option>
            <option value="L">Latency</option>
            <option value="LF">Load Factor</option>
            <option value="MN">Maintainability</option>
            <option value="FT">Fault Tolerance</option>
            <option value="O">Operational</option>
            <option value="PE">Performance</option>
            <option value="PO">Portability</option>
            <option value="RE">Reliability</option>
            <option value="SC">Scalability</option>
            <option value="SE">Security</option>
            <option value="MD">Modularity</option>
            <option value="US">Usability</option>
            <option value="IN">Interoperability</option>
            <option value="AC">Accessibility</option>
            <option value="CO">Compatibility</option>
          </select>
        </div>
      </div>

    </div>
  </div>
`;

  });

  new bootstrap.Modal(
    document.getElementById("nfrConfirmModal")
  ).show();
}

// ✅ FIXED: Submit NFR Confirmation with proper error display
async function submitNFRConfirmation() {
  if (!extractedData || !extractedData.project_id) {
    showErrorModal("Project ID is missing. Cannot save feedback.");
    return;
  }

  // ✅ Collect user selections
  const items = [];
  document.querySelectorAll(".nfr-user-choice").forEach(sel => {
    items.push({
      description: sel.dataset.desc,
      type: sel.value
    });
  });

  // ✅ Validate: all NFRs must have a type selected
  if (items.some(i => !i.type)) {
    showNfrInlineError("⚠️ Please select a type for all NFRs before continuing.");
    return;
  }

  try {
    /* 1️⃣ Hide error if validation passed */
    hideNfrInlineError();

    /* 2️⃣ Close NFR modal */
    const nfrModal = bootstrap.Modal.getInstance(
      document.getElementById("nfrConfirmModal")
    );
    if (nfrModal) {
      nfrModal.hide();
    }

    /* 3️⃣ Show loading screen */
    pendingConfirmation = true;

    document.getElementById('resultContent').classList.add('hidden');
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('loadingMessage').classList.remove('hidden');
    startLoadingAnimation();

    /* 4️⃣ Send confirmation to backend */
    const response = await fetch("/confirm_nfr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: extractedData.project_id,
        items: items
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to save confirmation");
    }

    const confirmData = await response.json();

    /* 5️⃣ Update extractedData with new results */
    extractedData.nfr_predictions = confirmData.nfr_predictions;
    extractedData.functional_method = confirmData.functional_method;
    extractedData.ordinal_method = confirmData.ordinal_method;
    extractedData.binary_method = confirmData.binary_method;
    extractedData.weighted_method = confirmData.weighted_method;
    extractedData.hybrid_method = confirmData.hybrid_method;
    
    /* 6️⃣ Short delay for UX */
    setTimeout(() => {
      stopLoadingAnimation();
      document.getElementById('loadingMessage').classList.add('hidden');

      pendingConfirmation = false;

      document.getElementById('progressSection').classList.remove('hidden');
      document.getElementById('resultContent').classList.remove('hidden');

      renderPhase(); // Show Phase 1 results
    }, 1200);

  } catch (error) {
    stopLoadingAnimation();
    document.getElementById('loadingMessage').classList.add('hidden');
    pendingConfirmation = false;

    console.error("NFR Confirmation Error:", error);
    showErrorModal(`Failed to save feedback: ${error.message}`);
  }
}

// ✅ FIXED: Helper functions for inline error display
function showNfrInlineError(message) {
  const box = document.getElementById("nfrInlineError");
  if (box) {
    box.innerText = message;
    box.classList.remove("d-none");
    
    // Scroll to error
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function hideNfrInlineError() {
  const box = document.getElementById("nfrInlineError");
  if (box) {
    box.classList.add("d-none");
  }
}

  function logout() {
    fetch("/logout", {
      method: "POST",
      credentials: "include"   // مهم عشان يمسح session
    })
    .then(() => {
      window.location.href = "/";
    })
    .catch(() => {
      window.location.href = "/";
    });
  }

  function logoutAnimated() {
    document.body.classList.add("page-exit");

    fetch("/logout", {
      method: "POST",
      credentials: "include"
    }).finally(() => {
      setTimeout(() => {
        window.location.href = "/Login?logout=1";

      }, 600);
    });
  }

  function toggleSideMenu() {
    document.getElementById("sideMenu").classList.toggle("open");
  }
  function backToDashboard(){
  document.getElementById("uploadView").classList.add("hidden");
  document.getElementById("dashboardView").classList.remove("hidden");
}
function showErrorModal(message) {

  // ✅ اقفلي NFR modal لو مفتوح
  const nfrModalEl = document.getElementById("nfrConfirmModal");
  const nfrModalInstance = bootstrap.Modal.getInstance(nfrModalEl);

  if (nfrModalInstance) {
    nfrModalInstance.hide();
  }

  // ✅ بعد كده افتحي error modal
  document.getElementById("errorModalMessage").innerText = message;

  const errorModal = new bootstrap.Modal(
    document.getElementById("errorModal"),
    { backdrop: "static" }
  );

  errorModal.show();
}
function showSrsVerifiedBadge() {
  const box = document.getElementById("srsVerificationStatus");

  if (!box) return;

  box.innerHTML = `
    <div class="alert alert-success d-flex align-items-center mt-3">
      <i class="bi bi-check-circle-fill me-2"></i>
      <strong>SRS Verified</strong>
      <span class="ms-2 text-muted">
        Functional and Non-Functional Requirements detected successfully.
      </span>
    </div>
  `;
}

function getProgressValue(phase) {
  return parseInt(phaseData[phase].pct.replace("%", ""));
}

async function syncProjectProgress() {
  if (!window.currentProjectId) return;

  const progress = getProgressValue(currentPhase);

  try {
    await fetch("/projects/update-progress", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        project_id: window.currentProjectId,
        phase: currentPhase,
        progress: progress
      })
    });
  } catch (e) {
    console.error("Failed to update project progress", e);
  }
}


function copyCode() {
  const code = document.getElementById("generatedCode").innerText;
  navigator.clipboard.writeText(code);
}

function downloadCode() {
  const code = document.getElementById("generatedCode").innerText;
  const blob = new Blob([code], { type: "text/plain" });

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "skeleton.js";
  a.click();
}

function regenerateCode() {
  alert("Regenerating code...");
}


function openProject(projectId) {
  console.log("📂 Opening project:", projectId);

  fetch(`/get_project/${projectId}`)
    .then(res => res.json())
    .then(data => {

      console.log("DATA:", data); // مهم للديباج

      if (data.error) {
        alert("Failed to load project");
        return;
      }

      // ======================
      // 1. Restore state
      // ======================
      extractedData = {
        functional: data.functional || [],
        nfr_predictions: data.nfr_predictions || [],
        functional_method: data.functional_method,
        ordinal_method: data.ordinal_method,
        binary_method: data.binary_method,
        weighted_method: data.weighted_method,
        hybrid_method: data.hybrid_method
      };

      currentPhase = data.current_phase || 1;
      selectedArchitecture = data.selectedArchitecture || null;

      window.currentProjectId = projectId;

      // ======================
      // 2. Switch UI
      // ======================
      document.getElementById('dashboardView').classList.add('hidden');
      document.getElementById('uploadView').classList.remove('hidden');

      // 🔥 أهم سطر (يخفي upload UI)
      document.getElementById("step-upload").classList.add("hidden");

      // ======================
      // 3. Show results UI
      // ======================
      document.getElementById('progressSection').classList.remove('hidden');
      document.getElementById('resultContent').classList.remove('hidden');

      // ======================
      // 4. Render correct phase
      // ======================
      renderPhase();

      console.log("✅ Project restored successfully");
    })
    .catch(err => {
      console.error("❌ Error loading project:", err);
    });
}  console.log("📂 Opening project:", projectId);

  fetch(`/get_project/${projectId}`)
    .then(res => res.json())
    .then(data => {

      if (data.error) {
        alert("Failed to load project");
        return;
      }

      // ======================
      // 1. Restore state
      // ======================
      extractedData = {
  functional: data.functional || [],
  nfr_predictions: data.nfr_predictions || [],
  functional_method: data.functional_method,
  ordinal_method: data.ordinal_method,
  binary_method: data.binary_method,
  weighted_method: data.weighted_method,
  hybrid_method: data.hybrid_method
};

currentPhase = data.current_phase || 1;
      selectedArchitecture = data.selectedArchitecture || null;

      window.currentProjectId = projectId;

      // ======================
      // 2. Switch UI
      // ======================
      document.getElementById('dashboardView').classList.add('hidden');
      document.getElementById('uploadView').classList.remove('hidden');

      // ======================
      // 3. Show results
      // ======================
      document.getElementById('progressSection').classList.remove('hidden');
      document.getElementById('resultContent').classList.remove('hidden');

      renderPhase();

      console.log("✅ Project loaded successfully");
    })
    .catch(err => {
      console.error("❌ Error loading project:", err);
    });