import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API Key Found:", bool(api_key))

if not api_key:
    print("GOOGLE_API_KEY not found in .env")
    exit()

genai.configure(api_key=api_key)

try:
    print("\nAvailable Models:\n")

    for model in genai.list_models():
        print(model.name)

except Exception as e:
    print("\nERROR:")
    print(e)