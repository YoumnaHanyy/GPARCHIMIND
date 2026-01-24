def convert_to_c4_plantuml(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml")
    lines.append("LAYOUT_WITH_LEGEND()")

    # 🔥 system is now a string
    system_name = arch["system"]

    lines.append(f'System_Boundary(system, "{system_name}") {{')

    # ---- Containers ----
    for c in arch.get("components", []):
        name = c["name"]
        desc = c.get("responsibility", "")
        tech = c.get("technology", "Service")
        alias = name.replace(" ", "")

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
        src = r["source"].replace(" ", "")
        dst = r["target"].replace(" ", "")
        rtype = r.get("type", "data-flow")

        if rtype == "event-flow":
            lines.append(f'Rel({src}, {dst}, "event", "Async")')
        else:
            lines.append(f'Rel({src}, {dst}, "call", "Sync")')

    lines.append("@enduml")
    return "\n".join(lines)
