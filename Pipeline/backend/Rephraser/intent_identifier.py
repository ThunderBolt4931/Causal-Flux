import os
import json
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
INTENT_PARAMS = os.path.join(base_dir, "intent_params.txt")

with open(INTENT_PARAMS, "r", encoding="utf-8") as f:
    system_prompt = f.read()

client = Client(api_key=os.getenv("GEMINI_API_KEY"))

def classify_intents(query: str):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                response_mime_type="application/json" 
            )
        )

        result = json.loads(response.text)
        
        return result.get("identified_intents", ["general inquiry"])

    except Exception as e:
        print("Intent classification failed:", e)
        return ["general query"]