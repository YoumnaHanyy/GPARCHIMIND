from ai.inference.predict_binary import predict_binary_architecture
from infrastructure.repositories.binary_repository import save_binary_result

def execute_binary_method():
    result = predict_binary_architecture()

    save_binary_result({
        "method": "binary",
        "result": result
    })

    return result
