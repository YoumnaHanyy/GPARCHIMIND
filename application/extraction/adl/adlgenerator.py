from ai.facades.architecture_facade import ArchitectureGenerationFacade

facade = ArchitectureGenerationFacade()

def generate_adl(req):
    return facade.generate_architecture(
        system_name=req["system_name"],
        functional_requirements=req["functional_requirements"],
        non_functional_requirements=req["non_functional_requirements"],
        architecture_style=req["architecture_style"]
    )