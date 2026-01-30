from infrastructure.repositories.human_feedback_repository import (
    load_confirmed_types,
    save_confirmed_types,
)
from ai.utils.nfr_mapping import get_nfr_label


def handle_user_confirmation(items: list):
    """
    Handle user confirmation of NFR types
    
    Args:
        items: [{"description": str, "type": str (NFR code)}]
    
    Returns:
        int: Number of confirmed items
    """
    confirmed_map = load_confirmed_types()
    feedback_rows = []

    for it in items:
        desc = (it.get("description") or "").strip()
        code = (it.get("type") or "").strip()

        if not desc or not code:
            continue

        # Store the code in the mapping
        confirmed_map[desc] = code
        
        # Get the label for display/logging
        label = get_nfr_label(code)
        
        feedback_rows.append({
            "title": it.get("title", ""),
            "description": desc,
            "type": code  # Store code, repository will add label
        })
        
        print(f"✅ Confirmed: {desc[:50]}... → {code} ({label})")

    save_confirmed_types(confirmed_map)

    return len(feedback_rows)