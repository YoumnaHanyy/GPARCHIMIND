from infrastructure.repositories.extraction_repository import ExtractionRepository
from ai.methods.functional_method import choose_architecture
from infrastructure.repositories.functional_repository import save_functional_method


def execute_functional_method(project_id: int):
    functional_reqs = ExtractionRepository.get_functional(project_id)

    if not functional_reqs:
        return {
            "top_architectures": [],
            "error": "No functional requirements found"
        }

    result = choose_architecture(functional_reqs)
    save_functional_method(project_id, result)

    return result