def convert_to_context_view(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml")
    lines.append("LAYOUT_WITH_LEGEND()")

    # 🔥 system is now a string
    system_name = arch["system"]

    # ---- Actor ----
    lines.append('Person(User, "End User", "Uses the system")')

    # ---- System ----
    lines.append(f'System(SystemA, "{system_name}", "System under design")')

    # ---- External Systems (inferred from components) ----
    for c in arch.get("components", []):
        lname = c["name"].lower()

        if "payment" in lname:
            lines.append('System_Ext(PaymentGateway, "Payment Gateway", "External payment processor")')
            lines.append('Rel(SystemA, PaymentGateway, "Processes payments via")')

        if "notification" in lname:
            lines.append('System_Ext(NotificationService, "Notification Service", "External notification system")')
            lines.append('Rel(SystemA, NotificationService, "Sends notifications via")')

    # ---- User Interaction ----
    lines.append('Rel(User, SystemA, "Uses")')

    lines.append("@enduml")
    return "\n".join(lines)
