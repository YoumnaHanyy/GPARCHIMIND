// ===============================
// ADL GENERATOR JS
// ===============================


// ENABLE BUTTON
function checkADLInputs() {

  const file =
    document.getElementById("adlFileInput").files.length;

  const architecture =
    document.getElementById("adlArchitecture").value;

  const btn =
    document.getElementById("generateAdlBtn");

  btn.disabled = !(file && architecture);
}



// GENERATE ADL
async function generateStandaloneADL() {

  const fileInput =
    document.getElementById("adlFileInput");

  const architecture =
    document.getElementById("adlArchitecture").value;

  if (!fileInput.files.length || !architecture) {

    alert(
      "Please upload SRS file and select architecture"
    );

    return;
  }

  try {

    const btn =
      document.getElementById("generateAdlBtn");

    btn.disabled = true;

    btn.innerText =
      "Generating...";


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
      throw new Error("Generation failed");
    }


    const blob =
      await response.blob();

    const url =
      window.URL.createObjectURL(blob);


// OPEN PDF IN MODAL
document.getElementById(
  "pdfViewer"
).src = url;


// OPEN MODAL
const pdfModal =
  new bootstrap.Modal(
    document.getElementById("pdfModal")
  );

pdfModal.show();

  } catch (err) {

    console.error(err);

    alert(
      "Error generating ADL PDF"
    );

  } finally {

    const btn =
      document.getElementById("generateAdlBtn");

    btn.disabled = false;

    btn.innerText =
      "Generate ADL";
  }
}