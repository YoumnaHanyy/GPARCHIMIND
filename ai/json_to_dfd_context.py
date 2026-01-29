
def convert_to_dfd_context(adl):
    lines = []
    lines.append("@startuml")
    lines.append("left to right direction")

    # Styles
    lines.append("skinparam rectangle {")
    lines.append("  BackgroundColor LightGray")
    lines.append("  BorderColor Black")
    lines.append("}")
    lines.append("skinparam package {")
    lines.append("  BackgroundColor White")
    lines.append("}")

    system_name = adl["system"]

    # ---- External Entities ----
    lines.append('rectangle "End User" <<external>>')
    
    # ---- Main System (Single Process) ----
    lines.append(f'rectangle "{system_name}" <<process>>')

    # ---- Basic User Data Flows ----
    lines.append(f'"End User" --> "{system_name}" : Requests / Inputs')
    lines.append(f'"{system_name}" --> "End User" : Responses / Results')

    # ---- External Systems inferred from ADL ----
    for s in adl.get("services", []):
        lname = s["name"].lower()

        if "payment" in lname:
            lines.append('rectangle "Payment Gateway" <<external>>')
            lines.append(f'"{system_name}" --> "Payment Gateway" : Payment Data')
            lines.append(f'"Payment Gateway" --> "{system_name}" : Payment Confirmation')

        if "notification" in lname:
            lines.append('rectangle "Notification Service" <<external>>')
            lines.append(f'"{system_name}" --> "Notification Service" : Notification Data')

    lines.append("@enduml")
    return "\n".join(lines)
