def convert_to_deployment_view(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!theme plain")
    lines.append("left to right direction")

    style = arch.get("style", "").lower()
    availability = arch.get("production_intents", {}).get("deployment", {}).get("availability", "")

    # -------- Client --------
    lines.append('node "Client" {')
    lines.append('  component "User"')
    lines.append('}')

    # -------- Ingress --------
    lines.append('node "Kubernetes Ingress" {')
    lines.append('  component "Ingress Controller"')
    lines.append('}')

    # -------- Classify components --------
    app_services = []
    broker_services = []
    db_services = []

    for c in arch.get("components", []):
        lname = c["name"].lower()
        if "database" in lname:
            db_services.append(c["name"])
        elif "broker" in lname or "queue" in lname:
            broker_services.append(c["name"])
        else:
            app_services.append(c["name"])

    # -------- Application Pods --------
    for s in app_services:
        lines.append(f'node "{s} Pod" {{')
        lines.append(f'  component "{s}"')
        lines.append('}')

    # -------- Broker Pods --------
    for b in broker_services:
        lines.append('node "Message Broker Pod" {')
        lines.append(f'  component "{b}"')
        lines.append('}')

    # -------- Database --------
    for d in db_services:
        lines.append('node "Database StatefulSet" {')
        lines.append(f'  database "{d}"')
        lines.append('}')

    # -------- Connections --------
    lines.append('"User" --> "Ingress Controller" : TCP/IP')
    lines.append('"Ingress Controller" --> "API Gateway Pod" : HTTP/HTTPS')
    # API Gateway routes to services

   # -------- Connections --------
    lines.append('"User" --> "Ingress Controller"')
    lines.append('"Ingress Controller" --> "API Gateway Pod"')

# API Gateway routes to services
    for s in app_services:
     if s != "API Gateway":
        lines.append(f'"API Gateway Pod" --> "{s} Pod"')

    for b in broker_services:
        for s in app_services:
            lines.append(f'"{s} Pod" --> "Message Broker Pod"')

    for d in db_services:
        for s in app_services:
            lines.append(f'"{s} Pod" --> "Database StatefulSet"')

    # -------- Multi-Region --------
    if "multi" in availability:
        lines.append('node "Secondary Region" {')
        lines.append('  component "Replica Services"')
        lines.append('}')
        lines.append('"Ingress Controller" --> "Secondary Region" : failover')

    lines.append("@enduml")
    return "\n".join(lines)
