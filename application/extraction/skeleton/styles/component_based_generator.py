from ai.inference.file_generator import (
    generate_component_based_project_files,
    generate_nfr_files,
    generate_pattern_files,
    get_main_file
)


def generate_component_based(functional, nfrs, patterns,
    language="python"):

    code = f"""
COMPONENT_BASED/
│

└── {get_main_file(language)}
"""

    requirements = []

    for fr in functional:

        if isinstance(fr, dict):

            desc = fr.get("description", "")

            if desc:
                requirements.append(desc)

    # FUNCTIONAL REQUIREMENTS

    try:

        parsed = generate_component_based_project_files(requirements,language)

        for folder, files in parsed.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("COMPONENT BASED ERROR:", e)

    # NFRs

    try:

        nfr_result = generate_nfr_files(nfrs, language)

        for folder, files in nfr_result.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("COMPONENT BASED NFR ERROR:", e)

    # DESIGN PATTERNS

    try:

        pattern_result = generate_pattern_files(
            patterns,
            requirements,
            language
        )

        patterns_folder = pattern_result.get("patterns", {})

        code += "\n\n├── patterns/\n"

        for pattern_name, files in patterns_folder.items():

            code += f"│   ├── {pattern_name}/\n"

            for file in files:

                code += f"│   │   ├── {file}\n"

    except Exception as e:

        print("COMPONENT BASED PATTERN ERROR:", e)

    return code