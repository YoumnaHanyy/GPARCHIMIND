import re

def alias_of(name: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return clean if clean else "Comp"


def convert_to_process_view(adl):
    lines = []
    lines.append("@startuml")
    lines.append("autonumber")

    # Actor
    lines.append("actor User")

    components = adl.get("components", [])
    runtime_flow = adl.get("runtime_flow", [])

    # -------- Fallback --------
    if not runtime_flow:
        lines.append("participant API")
        lines.append("participant Service")
        lines.append("participant Database")
        lines.append("API -> Service : processRequest()")
        lines.append("Service -> Database : storeData()")
        lines.append("@enduml")
        return "\n".join(lines)

    # -------- Participants --------
    alias_map = {}

    for c in components:
        name = c["name"]
        alias = alias_of(name)
        alias_map[name] = alias
        lines.append(f'participant "{name}" as {alias}')

    # -------- Runtime Flow --------
    for step in runtime_flow:
        src_name = step["from"]
        dst_name = step["to"]
        action = step["action"]
        mode = step.get("mode", "sync")

        src = alias_map.get(src_name, alias_of(src_name))
        dst = alias_map.get(dst_name, alias_of(dst_name))

        if mode == "async":
            lines.append(f'{src} ->> {dst} : {action}')
        else:
            lines.append(f'{src} -> {dst} : {action}')

    lines.append("@enduml")
    return "\n".join(lines)
