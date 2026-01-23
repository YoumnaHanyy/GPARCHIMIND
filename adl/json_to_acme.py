def normalize(name: str) -> str:
    return name.replace(" ", "")

def convert_to_acme(adl):
    lines = []
    system_name = normalize(adl["system"]["name"])

    # ================= SYSTEM =================
    lines.append(f"System {system_name} {{")
    lines.append('  Property view : String = "runtime";')

    # ================= COMPONENTS =================
    for s in adl.get("services", []):
        comp_name = normalize(s["name"])
        responsibility = s.get("responsibility", "")

        # ---- heuristics for production semantics ----
        name_lower = s["name"].lower()

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

        if "loadbalancer" in name_lower:
            lines.append('    Property type : String = "L7";')
            lines.append('    Property healthChecks : Boolean = true;')

        if "queue" in name_lower or "broker" in name_lower:
            lines.append('    Property messaging : String = "asynchronous";')

        if "database" in name_lower:
            lines.append('    Property consistency : String = "eventual";')
            lines.append('    Property replication : String = "multi-node";')

        lines.append("  }")

    # ================= ATTACHMENTS =================
    for r in adl.get("relationships", []):
        src = normalize(r.get("source"))
        dst = normalize(r.get("target"))
        rtype = r.get("type", "data-flow")

        lines.append(f"  Attachment {src}.out to {dst}.in {{")

        # runtime semantics
        if rtype == "event-flow":
            lines.append('    Property delivery : String = "at-least-once";')
            lines.append('    Property ordering : Boolean = false;')
        else:
            lines.append('    Property delivery : String = "at-least-once";')
            lines.append('    Property ordering : Boolean = true;')

        lines.append('    Property backpressure : String = "bounded";')
        lines.append("  }")

    # ================= RESILIENCE =================
    lines.append('  Property resilience_pattern : String = "Retry + CircuitBreaker";')
    lines.append('  Property failure_isolation : Boolean = true;')

    # ================= SECURITY =================
    lines.append('  Property authentication : String = "service-to-service";')
    lines.append('  Property authorization : String = "RBAC";')
    lines.append('  Property encryption : String = "in-transit";')

    # ================= QUALITY ATTRIBUTES =================
    qa = adl.get("qualityAttributes", {})
    for k, v in qa.items():
        lines.append(f'  Property constraint_{k} : String = "{v}";')

    lines.append("}")
    return "\n".join(lines)
