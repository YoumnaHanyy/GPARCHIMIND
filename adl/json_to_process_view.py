def convert_to_process_view(adl):
    lines = []
    lines.append("@startuml")
    lines.append("autonumber")

    print("PROCESS VIEW ADL KEYS:", adl.keys())
    print("RUNTIME FLOW:", adl.get("runtime_flow"))

    # Actor
    lines.append("actor User")

    components = adl.get("components", [])
    runtime_flow = adl.get("runtime_flow", [])

    # لو مفيش runtime_flow نعمل fallback رسم بسيط
    if not runtime_flow:
        lines.append("participant API")
        lines.append("participant Service")
        lines.append("participant Database")
        lines.append("API -> Service : processRequest()")
        lines.append("Service -> Database : storeData()")
        lines.append("@enduml")
        return "\n".join(lines)

    # Participants من components
    for c in components:
        name = c["name"]
        alias = name.replace(" ", "")
        lines.append(f'participant "{name}" as {alias}')

    # Runtime flow
    for step in runtime_flow:
        src = step["from"].replace(" ", "")
        dst = step["to"].replace(" ", "")
        action = step["action"]
        mode = step.get("mode", "sync")

        if mode == "async":
            lines.append(f'{src} ->> {dst} : {action}')
        else:
            lines.append(f'{src} -> {dst} : {action}')

    lines.append("@enduml")
    return "\n".join(lines)
