// ===============================
// ADL DASHBOARD JS
// ===============================

function checkADLInputs() {

  const file =
    document.getElementById("adlFileInput").files.length;

  const arch =
    document.getElementById("adlArchitecture").value;

  const btn =
    document.getElementById("generateAdlBtn");

  btn.disabled = !(file && arch);
}


// ===============================
// GENERATE ADL PDF
// ===============================

async function generateStandaloneADL() {

  const fileInput =
    document.getElementById("adlFileInput");

  const architecture =
    document.getElementById("adlArchitecture").value;

  if (!fileInput.files.length || !architecture) {
    alert("Please upload SRS file and select architecture");
    return;
  }

  try {

    const btn =
      document.getElementById("generateAdlBtn");

    btn.disabled = true;
    btn.innerText = "Generating...";

    const formData = new FormData();

    formData.append(
      "file",
      fileInput.files[0]
    );

    formData.append(
      "architecture",
      architecture
    );

    const response = await fetch(
      "/adl/generate-pdf",
      {
        method: "POST",
        body: formData
      }
    );

    if (!response.ok) {
      throw new Error("Failed to generate ADL");
    }

    const blob = await response.blob();

    const url =
      window.URL.createObjectURL(blob);

    const a =
      document.createElement("a");

    a.href = url;
    a.download = "Architecture_Report.pdf";

    document.body.appendChild(a);

    a.click();

    a.remove();

    window.URL.revokeObjectURL(url);

    // refresh projects after generation
    setTimeout(() => {
      location.reload();
    }, 1500);

  } catch (err) {

    console.error(err);

    alert(
      "Error generating ADL PDF"
    );

  } finally {

    const btn =
      document.getElementById("generateAdlBtn");

    btn.disabled = false;
    btn.innerText = "Generate ADL";
  }
}


// ===============================
// OPEN PROJECT
// ===============================

function openProject(projectId) {


  

  window.location.href =
    `/adl-project/${projectId}`;
}
async function toggleStar(projectId, event) {

    event.stopPropagation();

    const icon = event.currentTarget.querySelector("i");

    try {

        const response = await fetch(
            `/project/${projectId}/toggle-star`,
            { method: "POST" }
        );

        const data = await response.json();

        if (data.starred) {

            icon.classList.remove("bi-bookmark");
            icon.classList.add("bi-bookmark-fill");

        } else {

            icon.classList.remove("bi-bookmark-fill");
            icon.classList.add("bi-bookmark");
        }

        event.currentTarget.classList.add("bookmark-bounce");

        setTimeout(() => {
            event.currentTarget.classList.remove("bookmark-bounce");
        }, 350);

    } catch (err) {
        console.error(err);
    }
}