import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

prompt = "Write a short sentence explaining why a photo of a cat is relevant to the query 'cute pet'."

try:
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    print("Gemini Response:", response.text)
except Exception as e:
    print("Error calling Gemini API:", e)
