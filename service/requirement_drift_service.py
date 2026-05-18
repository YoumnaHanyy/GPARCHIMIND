import json
from service.srs_extractor import generate


class RequirementDriftService:

    @staticmethod
    def detect_changes(
        old_frs,
        old_nfrs,
        new_frs,
        new_nfrs
    ):

        prompt = f"""
You are an AI software architecture analyst.

Compare OLD requirements with NEW requirements.

Detect:
- Added Functional Requirements
- Removed Functional Requirements
- Edited Functional Requirements
- Added Non Functional Requirements
- Removed Non Functional Requirements
- Edited Non Functional Requirements

Return ONLY valid JSON:

{{
  "fr_added": [],
  "fr_removed": [],
  "fr_edited": [],
  "nfr_added": [],
  "nfr_removed": [],
  "nfr_edited": [],
  "summary": ""
}}

OLD FUNCTIONAL:
{json.dumps(old_frs, indent=2)}

NEW FUNCTIONAL:
{json.dumps(new_frs, indent=2)}

OLD NON FUNCTIONAL:
{json.dumps(old_nfrs, indent=2)}

NEW NON FUNCTIONAL:
{json.dumps(new_nfrs, indent=2)}
"""

        output = generate(prompt)

        return json.loads(output)