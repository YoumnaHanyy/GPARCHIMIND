def build_prompt(frs_text, nfrs_text):
    return f"""
You are a senior software architect.

Your task is to analyze system requirements and assign a confidence score (0.0 to 1.0) for each architectural feature.

----------------------------------------
RULES:

- Do NOT hallucinate features.
- Base your decision on the requirements text.
- HOWEVER, you ARE ALLOWED to use reasonable architectural inference.

- If multiple weak signals exist → combine them.
- Do NOT be overly conservative.
- Slight overestimation is acceptable if logically justified.

----------------------------------------
IMPORTANT DISTINCTIONS:

- "Fast", "quick response", "2 seconds" → LOW_LATENCY (NOT REAL_TIME)
- Notifications + system reactions → may indicate EVENT_DRIVEN
- Integration with APIs or external systems → may indicate DISTRIBUTED_SYSTEM
- "Reliable" or "availability" → FAULT_TOLERANCE
- "Maintainable" → HIGH_MAINTAINABILITY
- "Scalable", "many users", "concurrent users" → HIGH_SCALABILITY

----------------------------------------
SCORING GUIDE:

- 0.9 – 1.0 → strong explicit evidence
- 0.7 – 0.89 → strong indirect evidence
- 0.4 – 0.69 → weak but reasonable signals
- 0.0 – 0.3 → no meaningful support

- If multiple indicators support a feature → increase score

----------------------------------------
FEATURES TO SCORE:

EVENT_DRIVEN  
REAL_TIME  
HIGH_SCALABILITY  
LOW_LATENCY  
HIGH_EXTENSIBILITY  
HIGH_MAINTAINABILITY  
FLEXIBLE_CREATION  
DYNAMIC_BEHAVIOR  
DISTRIBUTED_SYSTEM  
FAULT_TOLERANCE  
HIGH_SECURITY  
MODULARITY  
HIGH_COUPLING_RISK  

----------------------------------------
OUTPUT FORMAT (STRICT):

Return ONLY valid JSON.
No explanation.
No extra text.

Example:

{{
  "EVENT_DRIVEN": 0.6,
  "REAL_TIME": 0.5,
  "HIGH_SCALABILITY": 0.9,
  "LOW_LATENCY": 0.7,
  "HIGH_EXTENSIBILITY": 0.3,
  "HIGH_MAINTAINABILITY": 0.6,
  "FLEXIBLE_CREATION": 0.2,
  "DYNAMIC_BEHAVIOR": 0.6,
  "DISTRIBUTED_SYSTEM": 0.7,
  "FAULT_TOLERANCE": 0.8,
  "HIGH_SECURITY": 0.9,
  "MODULARITY": 0.7,
  "HIGH_COUPLING_RISK": 0.2
}}

----------------------------------------
REQUIREMENTS:

{frs_text}

{nfrs_text}

JSON:
"""