import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


def convert_to_context_view(arch):
    lines = []
    lines.append("@startuml")
    lines.append("!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml")
    lines.append("LAYOUT_WITH_LEGEND()")

    system_name = arch["system"]
    system_alias = alias_of(system_name)

    # ---- Actor ----
    user_alias = "UserActor"
    lines.append(f'Person({user_alias}, "End User", "Uses the system")')

    # ---- System ----
    lines.append(f'System({system_alias}, "{system_name}", "System under design")')

    # ---- External Systems (derived from components) ----
    declared_ext = set()

    for c in arch.get("components", []):
        lname = c["name"].lower()

        if "payment" in lname and "payment" not in declared_ext:
            ext_name = "Payment Gateway"
            ext_alias = alias_of(ext_name)
            declared_ext.add("payment")

            lines.append(
                f'System_Ext({ext_alias}, "{ext_name}", "External payment processor")'
            )
            lines.append(
                f'Rel({system_alias}, {ext_alias}, "Processes payments via")'
            )

        if "notification" in lname and "notification" not in declared_ext:
            ext_name = "Notification Service"
            ext_alias = alias_of(ext_name)
            declared_ext.add("notification")

            lines.append(
                f'System_Ext({ext_alias}, "{ext_name}", "External notification system")'
            )
            lines.append(
                f'Rel({system_alias}, {ext_alias}, "Sends notifications via")'
            )

    # ---- User Interaction ----
    lines.append(f'Rel({user_alias}, {system_alias}, "Uses")')

    lines.append("@enduml")
    return "\n".join(lines)
