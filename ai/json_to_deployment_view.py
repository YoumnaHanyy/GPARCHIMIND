import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


def convert_to_deployment_view(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!theme plain")
    lines.append("left to right direction")

    availability = (
        arch.get("production_intents", {})
            .get("deployment", {})
            .get("availability", "")
            .lower()
    )

    # -------- Client --------
    client_alias = "Client"
    lines.append(f'node "{client_alias}" {{')
    lines.append('  component "User"')
    lines.append('}')

    # -------- Ingress --------
    ingress_alias = "IngressController"
    lines.append(f'node "Kubernetes Ingress" as {ingress_alias} {{')
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
    pod_aliases = {}

    for s in app_services:
        pod_name = f"{s} Pod"
        pod_alias = alias_of(pod_name)
        pod_aliases[s] = pod_alias

        lines.append(f'node "{pod_name}" as {pod_alias} {{')
        lines.append(f'  component "{s}"')
        lines.append('}')

    # -------- Broker Pods --------
    broker_alias = None
    if broker_services:
        broker_alias = "MessageBrokerPod"
        lines.append(f'node "Message Broker Pod" as {broker_alias} {{')
        for b in broker_services:
            lines.append(f'  component "{b}"')
        lines.append('}')

    # -------- Database --------
    db_alias = None
    if db_services:
        db_alias = "DatabaseStatefulSet"
        lines.append(f'node "Database StatefulSet" as {db_alias} {{')
        for d in db_services:
            lines.append(f'  database "{d}"')
        lines.append('}')

    # -------- Connections --------
    lines.append(f'{client_alias} --> {ingress_alias} : HTTP')

    # Ingress to API Gateway (only if exists)
    if "API Gateway" in app_services:
        gw_alias = pod_aliases["API Gateway"]
        lines.append(f'{ingress_alias} --> {gw_alias} : HTTP')

        # Gateway to other services
        for s, alias in pod_aliases.items():
            if s != "API Gateway":
                lines.append(f'{gw_alias} --> {alias}')

    # Services to broker
    if broker_alias:
        for alias in pod_aliases.values():
            lines.append(f'{alias} --> {broker_alias}')

    # Services to database
    if db_alias:
        for alias in pod_aliases.values():
            lines.append(f'{alias} --> {db_alias}')

    # -------- Multi-Region --------
    if "multi" in availability:
        lines.append('node "Secondary Region" as SecondaryRegion {')
        lines.append('  component "Replica Services"')
        lines.append('}')
        lines.append(f'{ingress_alias} --> SecondaryRegion : failover')

    lines.append("@enduml")
    return "\n".join(lines)
