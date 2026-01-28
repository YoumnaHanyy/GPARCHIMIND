import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


def convert_to_c4_plantuml(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml")
    lines.append("LAYOUT_WITH_LEGEND()")

    system_name = arch["system"]
    system_alias = alias_of(system_name)

    lines.append(f'System_Boundary({system_alias}, "{system_name}") {{')

    # ---- Containers ----
    for c in arch.get("components", []):
        name = c["name"]
        desc = c.get("responsibility", "")
        tech = c.get("technology", "Service")

        alias = alias_of(name)
        lname = name.lower()

        if "database" in lname:
            lines.append(
                f'  ContainerDb({alias}, "{name}", "Database", "{desc}")'
            )
        elif "broker" in lname or "queue" in lname:
            lines.append(
                f'  ContainerQueue({alias}, "{name}", "Message Broker", "{desc}")'
            )
        elif "gateway" in lname or "api" in lname:
            lines.append(
                f'  Container({alias}, "{name}", "REST API", "{desc}")'
            )
        else:
            lines.append(
                f'  Container({alias}, "{name}", "{tech}", "{desc}")'
            )

    lines.append("}")

    # ---- Relationships ----
    for r in arch.get("relationships", []):
        src = alias_of(r["source"])
        dst = alias_of(r["target"])
        rtype = r.get("type", "data-flow")

        if rtype == "event-flow":
            lines.append(f'Rel({src}, {dst}, "event", "Async")')
        else:
            lines.append(f'Rel({src}, {dst}, "call", "Sync")')

    lines.append("@enduml")
    return "\n".join(lines)
