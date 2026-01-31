def convert_to_acme(adl):
    lines = []
    system_name = adl["system"]["name"]

    lines.append(f"System {system_name} {{")

    # Components
    for s in adl.get("services", []):
        lines.append(
            f"  Component {s['name']} = new Component {{}}"
        )

    # Relationships (SAFE)
    for r in adl.get("relationships", []):
        src = r.get("from")
        dst = r.get("to")

        if not src or not dst:
            continue  # skip invalid relationships

        lines.append(
            f"  Attachment {src}.out to {dst}.in;"
        )

    lines.append("}")
    return "\n".join(lines)
