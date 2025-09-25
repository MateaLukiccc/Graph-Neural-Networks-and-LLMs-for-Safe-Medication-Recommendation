import pymupdf
import base64
from models import ImageResponse
from parse_llm import client, IMAGE_PARSE_PROMPT, IMAGE_PARSE_MODEL
import json


def extract_admissions(text, names=False):
    """
    Extracts ICD9_CODE, PROCEDURE, and ATC3 codes for each admission in the text.
    Returns a list of dicts, each representing an admission.
    """
    lines = text.splitlines()
    admissions = []
    current = None
    i = 0
    icd_line = "ICD9 Diagnosis"
    proc_line = "Procedures"
    atc3_line = "Medications (ATC3)"
    
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Admission #"):
            if current:
                admissions.append(current)
            current = {"ICD9_CODE": [], "PROCEDURE": [], "ATC3": []}
        elif line == icd_line and current is not None:
            i += 1
            if i < len(lines):
                current["ICD9_CODE"] = [code.strip() for code in lines[i].split(",") if code.strip()]
        elif line == proc_line and current is not None:
            i += 1
            if i < len(lines):
                current["PROCEDURE"] = [code.strip() for code in lines[i].split(",") if code.strip()]
        elif line == atc3_line and current is not None:
            i += 1
            if i < len(lines):
                current["ATC3"] = [code.strip() for code in lines[i].split(",") if code.strip()]
        i += 1
    if current:
        admissions.append(current)
    return admissions


def parse_pdf(file_path, names=False):
    """
    Parses a PDF file and extracts ICD9_CODE, PROCEDURE, and ATC3 codes.
    """
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return extract_admissions(text, names=names)


def parse_image(file_path):
    """
    Sends an image from file_path to the Groq API and retrieves the response.
    """
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    completion = client.chat.completions.create(
        model=IMAGE_PARSE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_PARSE_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "product_review",
                "schema": ImageResponse.model_json_schema()
            }
        }
    )
    return completion.choices[0].message


if __name__ == "__main__":
    pdf_path = "example_data/example.pdf"
    admissions = parse_pdf(pdf_path, names=True)
    for idx, adm in enumerate(admissions, 1):
        print(f"Admission #{idx}:")
        print("  ICD9_CODE:", adm["ICD9_CODE"])
        print("  PROCEDURE:", adm["PROCEDURE"])
        print("  ATC3:", adm["ATC3"])
        print()
        
    image_path = "example_data/example_codes.jpeg"
    response = parse_image(image_path)
    parsed_data = json.loads(response.content)
    image_response = ImageResponse(**parsed_data)

    for idx, adm in enumerate(image_response.admissions, 1):
        print(f"Admission #{idx}:")
        print("  ICD9_CODE:", adm.icd)
        print("  PROCEDURE:", adm.procedure)
        print("  ATC3:", adm.atc3)
        print()