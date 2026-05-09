# application/extraction/skeleton/styles/pipe_filter_generator.py

from ai.inference.file_generator import (
    generate_pipe_filter_project_files,
    generate_nfr_files,
    generate_pattern_files
)


def generate_pipe_filter(functional, nfrs, patterns):

    code = """
PIPE_FILTER/
│

└── main.py
"""

    requirements = []

    for fr in functional:

        if isinstance(fr, dict):

            desc = fr.get("description", "")

            if desc:
                requirements.append(desc)

    # FUNCTIONAL REQUIREMENTS

    try:

        parsed = generate_pipe_filter_project_files(requirements)

        for folder, files in parsed.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("PIPE FILTER ERROR:", e)

    # NFRs

    try:

        nfr_result = generate_nfr_files(nfrs)

        for folder, files in nfr_result.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("PIPE FILTER NFR ERROR:", e)

    # DESIGN PATTERNS

    try:

        pattern_result = generate_pattern_files(
            patterns,
            requirements
        )

        patterns_folder = pattern_result.get("patterns", {})

        code += "\n\n├── patterns/\n"

        for pattern_name, files in patterns_folder.items():

            code += f"│   ├── {pattern_name}/\n"

            for file in files:

                code += f"│   │   ├── {file}\n"

    except Exception as e:

        print("PIPE FILTER PATTERN ERROR:", e)

    return code