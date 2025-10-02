import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from typing import Optional


load_dotenv()

class DrugReport(BaseModel):
    high_confidence_drugs: str
    medium_confidence_drugs: str
    low_confidence_drugs: str


class DrugReportGenerator:
    """
    A class for generating structured drug reports using Groq's LLM API.
    """

    REPORT_PARSE_PROMPT = """
You are a medical assistant that explains automated drug recommendation outputs for doctors.
You will be given:
- Patient diagnostic and procedural codes (ICD9, ATC3, procedures).
- A machine-generated drug recommendation report.

Your task:
1. Translate ATC codes into human-readable drug classes and common uses.
2. Organize recommendations into three groups: high confidence, medium confidence, low confidence.
3. For each drug class, explain in 1-3 sentences why it might have been suggested,
   referencing patient conditions or procedures when possible.
   Example: "A06A (laxatives) - suggested likely due to history of constipation (ICD9: 564.0)."
4. Keep the output structured and readable for clinicians who want quick insights.
5. Do not invent clinical recommendations beyond what the data supports.
6. Output results in JSON with the following keys:
   - high_confidence_drugs (list of explanations)
   - medium_confidence_drugs (list of explanations)
   - low_confidence_drugs (list of explanations)

Be concise but informative, so a physician can understand the reasoning at a glance.
"""
    REPORT_PARSE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client with API key.
        """
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in environment.")
        self.client = Groq(api_key=api_key)

    def generate_report(self, report_text: str) -> DrugReport:
        """
        Generate a structured drug report from raw text.
        """
        completion = self.client.chat.completions.create(
            model=self.REPORT_PARSE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.REPORT_PARSE_PROMPT},
                        {"type": "text", "text": report_text},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "drug_report",
                    "schema": DrugReport.model_json_schema(),
                },
            },
        )
        data = completion.choices[0].message.content
        return data
