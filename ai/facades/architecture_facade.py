from ai.ai_engine import ai_generate_architecture


class ArchitectureGenerationFacade:
    """
    Facade that provides a simplified interface
    to the AI-based architecture generation subsystem.
    """

    def generate_architecture(
        self,
        system_name: str,
        functional_requirements,
        non_functional_requirements,
        architecture_style: str
    ):
        return ai_generate_architecture(
            system=system_name,
            frs=functional_requirements,
            nfrs=non_functional_requirements,
            style=architecture_style
        )
