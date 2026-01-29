# ai/utils/nfr_mapping.py

NFR_MAP = {
    "Performance": "PE",
    "Scalability": "SC",
    "Maintainability": "MN",
    "Availability": "A",
    "Security": "SE",
    "Usability": "US",
    "Portability": "PO",
    "Operational ": "O",
    "Load Factor": "LF",
    "Latency": "L",
}

# Reverse mapping: abbreviation -> full name
NFR_MAP_REVERSE = {v: k for k, v in NFR_MAP.items()}
