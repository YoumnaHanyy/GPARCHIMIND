import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


def convert_to_dfd_context(adl):
    lines = []
    lines.append("@startuml")
    lines.append("left to right direction")

    # ---- Styles ----
    lines.append("skinparam rectangle {")
    lines.append("  BackgroundColor LightGray")
    lines.append("  BorderColor Black")
    lines.append("}")
    lines.append("skinparam package {")
    lines.append("  BackgroundColor White")
    lines.append("}")

    system_name = adl["system"]
    system_alias = alias_of(system_name)

    # ---- External Entity: User ----
    user_alias = "EndUser"
    lines.append(f'rectangle "{user_alias}" as {user_alias} <<external>>')

    # ---- Main Process (System) ----
    lines.append(f'rectangle "{system_name}" as {system_alias} <<process>>')

    # ---- User Data Flows ----
    lines.append(f'{user_alias} --> {system_alias} : Requests / Inputs')
    lines.append(f'{system_alias} --> {user_alias} : Responses / Results')

    # ---- External Systems inferred from components ----
    declared_ext = set()

    for c in adl.get("components", []):
        lname = c["name"].lower()

        if "payment" in lname and "payment" not in declared_ext:
            ext_name = "Payment Gateway"
            ext_alias = alias_of(ext_name)
            declared_ext.add("payment")

            lines.append(f'rectangle "{ext_name}" as {ext_alias} <<external>>')
            lines.append(f'{system_alias} --> {ext_alias} : Payment Data')
            lines.append(f'{ext_alias} --> {system_alias} : Payment Confirmation')

        if "notification" in lname and "notification" not in declared_ext:
            ext_name = "Notification Service"
            ext_alias = alias_of(ext_name)
            declared_ext.add("notification")

            lines.append(f'rectangle "{ext_name}" as {ext_alias} <<external>>')
            lines.append(f'{system_alias} --> {ext_alias} : Notification Data')

    lines.append("@enduml")
    return "\n".join(lines)
