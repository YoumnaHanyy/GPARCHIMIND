from ai.inference.predict_binary import predict_binary_architecture
from infrastructure.repositories.binary_repository import save_binary_result

def execute_binary_method(project_id):
    result = predict_binary_architecture()

    save_binary_result({
        "project_id": project_id,
        "result": result
    })

    return result