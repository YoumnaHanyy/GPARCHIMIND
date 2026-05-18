from fastapi import APIRouter
from fastapi.responses import FileResponse
import tempfile
import zipfile
import os

router = APIRouter()


def parse_tree_to_files(tree_text):
    paths = []
    lines = tree_text.splitlines()
    
    # Stack stores (indent_level, folder_path)
    folder_stack = []
    current_root = None

    for line in lines:
        if not line.strip():
            continue

        # Detect root folder (e.g. "COMPONENT_BASED/")
        stripped = line.strip()
        if stripped.endswith("/") and stripped.replace("/", "").replace("_", "").isupper():
            current_root = stripped.rstrip("/")
            folder_stack = [(0, current_root)]
            continue

        if current_root is None:
            continue

        # Calculate indent level by counting leading spaces/tree chars
        # Remove tree drawing characters to get clean name
        clean_name = (
            line.replace("├──", "")
                .replace("└──", "")
                .replace("│", "")
                .replace("    ", "\t")  # normalize 4-spaces to tab
        )

        # Count leading tabs to determine depth
        depth = len(clean_name) - len(clean_name.lstrip("\t"))
        clean_name = clean_name.strip()

        if not clean_name:
            continue

        # Pop stack to current depth
        while folder_stack and folder_stack[-1][0] >= depth:
            folder_stack.pop()

        parent_path = folder_stack[-1][1] if folder_stack else current_root

        if clean_name.endswith("/"):
            # It's a folder
            folder_name = clean_name.rstrip("/")
            full_path = f"{parent_path}/{folder_name}"
            folder_stack.append((depth, full_path))
            paths.append(full_path + "/")
        else:
            # It's a file
            full_path = f"{parent_path}/{clean_name}"
            paths.append(full_path)

    return paths


@router.post("/download-skeleton")
async def download_skeleton(data: dict):

    tree = data.get("tree", "")

    paths = parse_tree_to_files(tree)

    temp_dir = tempfile.mkdtemp()

    for path in paths:

        full_path = os.path.join(temp_dir, path)

        folder = os.path.dirname(full_path)

        os.makedirs(folder, exist_ok=True)

        if "." in path:

          with open(full_path, "w") as f:
              f.write("")
        else:
          os.makedirs(full_path, exist_ok=True)

    zip_path = os.path.join(temp_dir, "code_skeleton.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:

        for root, dirs, files in os.walk(temp_dir):

            for file in files:

                if file == "code_skeleton.zip":
                    continue

                file_path = os.path.join(root, file)

                arcname = os.path.relpath(file_path, temp_dir)

                zipf.write(file_path, arcname)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="code_skeleton.zip"
    )