from ai.inference.file_generator import (
    generate_project_files,
    generate_nfr_files,
    generate_pattern_files
)

def generate_rest(functional, nfrs, patterns):

    code = """
REST_API/
│

└── main.py
"""

    requirements = []

    for fr in functional:

        if isinstance(fr, dict):

            desc = fr.get("description", "")

            if desc:
                requirements.append(desc)

    try:

        parsed = generate_project_files(requirements)

        for folder, files in parsed.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("ERROR:", e)
        # NFR AI generation

    try:

        nfr_result = generate_nfr_files(nfrs)

        for folder, files in nfr_result.items():

            code += f"\n\n├── {folder}/\n"

            for file in files:

                code += f"│   ├── {file}\n"

    except Exception as e:

        print("NFR ERROR:", e)
    
        # DESIGN PATTERNS AI GENERATION

    try:

        pattern_result = generate_pattern_files(patterns,requirements)

        patterns_folder = pattern_result.get("patterns", {})

        code += "\n\n├── patterns/\n"

        for pattern_name, files in patterns_folder.items():

           code += f"│   ├── {pattern_name}/\n"

           for file in files:

             code += f"│   │   ├── {file}\n"
    except Exception as e:

        print("PATTERN ERROR:", e)
    return code