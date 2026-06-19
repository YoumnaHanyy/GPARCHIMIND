def normalize(name: str) -> str:
    return name.replace(" ", "")


def convert_to_acme(arch):
    lines = []
    system_name = normalize(arch["system"])

    # ================= SYSTEM =================
    lines.append(f"System {system_name} {{")
    lines.append('  Property view : String = "runtime";')
    lines.append(f'  Property architectureStyle : String = "{arch.get("style", "")}";')
    lines.append(f'  Property source : String = "{arch.get("source", "")}";')

    # ================= COMPONENTS =================
    for c in arch.get("components", []):
        comp_name = normalize(c["name"])
        responsibility = c.get("responsibility", "")

        name_lower = c["name"].lower()

        # ---- heuristics for production semantics ----
        if "database" in name_lower:
            state = "stateful"
            scaling = "vertical"
        else:
            state = "stateless"
            scaling = "horizontal"

        lines.append(f"  Component {comp_name} = new Component {{")
        lines.append("    Port in;")
        lines.append("    Port out;")
        lines.append(f'    Property responsibility : String = "{responsibility}";')
        lines.append(f'    Property state : String = "{state}";')
        lines.append(f'    Property scaling : String = "{scaling}";')

        if "gateway" in name_lower or "loadbalancer" in name_lower:
            lines.append('    Property type : String = "L7";')
            lines.append('    Property healthChecks : Boolean = true;')

        if "queue" in name_lower or "broker" in name_lower:
            lines.append('    Property messaging : String = "asynchronous";')

        if "database" in name_lower:
            lines.append('    Property consistency : String = "eventual";')
            lines.append('    Property replication : String = "multi-node";')

        lines.append("  }")

    # ================= ATTACHMENTS =================
    for r in arch.get("relationships", []):
        src = normalize(r.get("source"))
        dst = normalize(r.get("target"))
        rtype = r.get("type", "data-flow")

        lines.append(f"  Attachment {src}.out to {dst}.in {{")

        if rtype == "event-flow":
            lines.append('    Property delivery : String = "at-least-once";')
            lines.append('    Property ordering : Boolean = false;')
        else:
            lines.append('    Property delivery : String = "at-least-once";')
            lines.append('    Property ordering : Boolean = true;')

        lines.append('    Property backpressure : String = "bounded";')
        lines.append("  }")

    # ================= RESILIENCE =================
    intents = arch.get("production_intents", {}).get("resilience", {})
    if intents:
        patterns = ", ".join(intents.get("patterns", []))
        lines.append(f'  Property resilience_pattern : String = "{patterns}";')
        lines.append(f'  Property delivery : String = "{intents.get("delivery", "")}";')
        lines.append(f'  Property ordering : Boolean = {str(intents.get("ordering", True)).lower()};')

    # ================= SECURITY =================
    security = arch.get("production_intents", {}).get("security", {})
    if security:
        lines.append(f'  Property authentication : String = "{security.get("authentication", "")}";')
        lines.append(f'  Property authorization : String = "{security.get("authorization", "")}";')

    lines.append("}")
    return "\n".join(lines)