from pydantic import BaseModel
from typing import List

class ImageResponse(BaseModel):
    admissions: List['Admission']

class Admission(BaseModel):
    icd: List[str]
    procedure: List[str]
    atc3: List[str]