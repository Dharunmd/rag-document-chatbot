import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API Key:", "FOUND" if api_key else "NOT FOUND")

if api_key:
    genai.configure(api_key=api_key)

    try:
        print("\nAvailable models:\n")
        for model in genai.list_models():
            print(model.name)
    except Exception as e:
        print("ERROR:", e)
