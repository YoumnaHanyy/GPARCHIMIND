# ai/utils/nfr_mapping.py

NFR_MAP = {
    "Availability": "A",
    "Fault Tolerance": "FT",
    "Latency": "L",
    "Load Factor": "LF",
    "Maintainability": "MA",
    "Manageability": "MN",
    "Operational": "O",
    "Performance": "PE",
    "Portability": "PO",
    "Reliability": "RE",
    "Scalability": "SC",
    "Security": "SE",
    "Modularity": "MD",
    "Usability": "US",
    "Interoperability": "IN",
    "Accessibility": "AC",
    "Compatibility": "CO"
}

# Reverse mapping: abbreviation -> full name
NFR_MAP_REVERSE = {v: k for k, v in NFR_MAP.items()}


def get_nfr_label(code: str) -> str:
    """
    Convert NFR code to full label
    Example: "PE" -> "Performance"
    
    Args:
        code: NFR abbreviation (e.g., "PE", "SE", "US")
    
    Returns:
        str: Full NFR label or the code itself if not found
    """
    return NFR_MAP_REVERSE.get(code, code)


def get_nfr_code(label: str) -> str:
    """
    Convert NFR label to code
    Example: "Performance" -> "PE"
    
    Args:
        label: Full NFR name (e.g., "Performance", "Security")
    
    Returns:
        str: NFR abbreviation or the label itself if not found
    """
    return NFR_MAP.get(label, label)