async function reevaluateArchitecture(projectId) {

    // =========================
    // Functional Requirements
    // =========================

    const frInputs =
        document.querySelectorAll(".fr-input");

    let frs = [];

    frInputs.forEach((input, index) => {

        if(input.value.trim() !== ""){

            frs.push({
                id: `FR-${index + 1}`,
                description: input.value.trim()
            });

        }

    });

    // =========================
    // Non Functional Requirements
    // =========================

    const nfrItems =
        document.querySelectorAll(".nfr-item");

    let nfrs = [];

    nfrItems.forEach((item) => {

        const type =
            item.querySelector(".nfr-type").value;

        const desc =
            item.querySelector(".nfr-description").value;

        if(type && desc.trim() !== ""){

            nfrs.push({
                title: type,
                description: desc.trim()
            });

        }

    });

    // =========================
    // SEND TO BACKEND
    // =========================

    const response = await fetch(

        `/project/${projectId}/reevaluate`,

        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                frs: frs,
                nfrs: nfrs
            })
        }
    );

    const data = await response.json();

    // =========================
    // SHOW RESULTS
    // =========================

    document.getElementById(
        "evaluationResult"
    ).innerHTML = `

        <div class="card border-0 shadow p-4">

            <h3 class="mb-4">

                Architecture Evolution Result

            </h3>

            <p>

                <strong>Previous Architecture:</strong>

                ${data.previous_architecture}

            </p>

            <p>

                <strong>Recommended Architecture:</strong>

                ${data.new_architecture}

            </p>

            <p>

                <strong>Change Needed:</strong>

                ${data.change_needed}

            </p>

        </div>

    `;
}