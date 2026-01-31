
from huggingface_hub import InferenceClient
import json
import re

HF_API_KEY = "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"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_API_KEY
)

def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("AI did not return valid JSON")
    return json.loads(match.group())

def ai_generate_architecture(frs, nfrs, style):
    prompt = f"""
You are designing the INTERNAL architecture of a tool called "ArchiMind".

IMPORTANT CONSTRAINTS:
- This is NOT an e-commerce system
- This is NOT a business application
- Do NOT invent domain-specific services like User, Order, Product, Payment, API Gateway
- All components MUST be directly derived from the given functional requirements

System Purpose:
AI-assisted software architecture selection and documentation.

Functional Requirements:
{frs}

Non-Functional Requirements:
{nfrs}

Architecture Style:
{style}

TASK:
1. Derive ONLY internal system components needed to fulfill the functional requirements
2. Each component must map clearly to one or more functional requirements
3. Use technical names related to architecture analysis and documentation ONLY

Return ONLY valid JSON in the following format:

{{
  "components": [
    {{
      "name": "ComponentName",
      "responsibility": "Clear technical responsibility"
    }}
  ],
  "relationships": [
    {{
      "source": "ComponentA",
      "target": "ComponentB",
      "type": "data-flow"
    }}
  ]
}}
"""

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a senior software architect."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2
    )

    text = response["choices"][0]["message"]["content"]
    return extract_json(text)
