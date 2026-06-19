"""Check Gemini API connectivity."""

import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def main():
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    print("GOOGLE_API_KEY:", "found" if api_key else "missing")

    if not api_key:
        return

    genai.configure(api_key=api_key)

    print("\nAvailable models:")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"- {model.name}")


if __name__ == "__main__":
    main()
