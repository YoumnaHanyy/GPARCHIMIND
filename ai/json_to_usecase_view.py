import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


def convert_to_usecase_view(arch):
    lines = []

    lines.append("@startuml")
    lines.append("left to right direction")
    lines.append("skinparam dpi 300")
    lines.append("skinparam packageStyle rectangle")

    system_name = arch["system"]

    # 👤 Actor
    lines.append('actor "End User" as user')

    # 🟦 System Boundary
    lines.append(f'rectangle "{system_name}" {{')

    usecases = []

    for c in arch.get("components", []):
        name = c["name"]
        resp = c.get("responsibility", "")

        label = resp if resp else f"Use {name}"
        label = (label[:40] + "...") if len(label) > 40 else label

        alias = f"{alias_of(name)}_{len(usecases)}"

        lines.append(f'  usecase "{label}" as {alias}')
        usecases.append(alias)

    lines.append("}")

    for uc in usecases:
        lines.append(f'user --> {uc}')

    lines.append("@enduml")

    return "\n".join(lines)