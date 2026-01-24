def convert_to_c4_plantuml(adl):
    lines = []
    lines.append("@startuml")
    lines.append("!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml")
    lines.append("LAYOUT_WITH_LEGEND()")

    system_name = adl["system"]["name"]
    lines.append(f'System_Boundary(system, "{system_name}") {{')

    for s in adl.get("services", []):
        name = s["name"]
        desc = s.get("responsibility", "")
        tech = s.get("technology", "Service")
        alias = name.replace(" ", "")
        lines.append(f'  Container({alias}, "{name}", "{tech}", "{desc}")')

    lines.append("}")

    for r in adl.get("relationships", []):
        src = r["source"].replace(" ", "")
        dst = r["target"].replace(" ", "")
        label = r.get("type", "")
        lines.append(f'Rel({src}, {dst}, "{label}")')

    lines.append("@enduml")
    return "\n".join(lines)
