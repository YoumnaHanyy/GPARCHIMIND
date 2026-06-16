import re

def alias_of(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)


