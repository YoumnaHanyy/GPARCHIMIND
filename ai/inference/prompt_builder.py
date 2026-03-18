def build_prompt(frs_text, nfrs_text):
    return f"""
You are a software architecture expert.

Extract features with a confidence score from 0 to 1.

Return ONLY JSON.

Example:
Input:
System sends real-time notifications and is scalable.

Output:
{{
  "EVENT_DRIVEN": 0.9,
  "REAL_TIME": 0.95,
  "HIGH_SCALABILITY": 0.9
}}

Features:
- EVENT_DRIVEN
- REAL_TIME
- HIGH_SCALABILITY
- LOW_LATENCY
- HIGH_EXTENSIBILITY
- HIGH_MAINTAINABILITY
- FLEXIBLE_CREATION
- DYNAMIC_BEHAVIOR
- DISTRIBUTED_SYSTEM
- FAULT_TOLERANCE
- HIGH_SECURITY
- MODULARITY
- HIGH_COUPLING_RISK

Requirements:
{frs_text}
{nfrs_text}

JSON:
"""