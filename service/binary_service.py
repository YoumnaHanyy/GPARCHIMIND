from ai.methods.binary_method import run_binary_method

from infrastructure.repositories.binary_repository import (
    save_binary_result
)

def execute_binary_method(
    project_id,
    binary_vector
):

    # ==========================================
    # RUN BINARY METHOD
    # ==========================================

    result = run_binary_method(
        binary_vector
    )

    print("BINARY RESULT:", result)

    # ==========================================
    # SAVE RESULT
    # ==========================================

    save_binary_result({

        "project_id":
            project_id,

        "result":
            result
    })

    return result