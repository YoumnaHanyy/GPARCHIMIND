def generate_code_skeleton(
    selected_architecture,
    functional_requirements,
    non_functional_requirements
):

    skeleton = f"""
Project Structure

Architecture:
{selected_architecture}

Folders:
- controllers/
- models/
- services/
- repositories/

Functional Modules:
"""

    for fr in functional_requirements:
        skeleton += f"\n- {fr}"

    skeleton += "\n\nNFR Support:\n"

    for nfr in non_functional_requirements:
        skeleton += f"\n- {nfr}"

    return skeleton