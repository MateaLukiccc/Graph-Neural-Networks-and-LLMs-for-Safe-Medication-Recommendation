from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

IMAGE_PARSE_PROMPT = "Read me the ICD9_CODE, PROCEDURE, and ATC3 codes for each admission in this image."
IMAGE_PARSE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))