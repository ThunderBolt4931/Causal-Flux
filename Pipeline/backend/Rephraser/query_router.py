import os
import json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAG_DOMAIN_DESCRIPTION = """
The available RAG database contains **Customer Service Transcripts** and **Business Analytics** data.
It includes information on:
- Agent performance (resolution rates, handle time, compliance).
- Customer complaints (refunds, double bookings, noise, amenities).
- Call metrics (escalations, sentiment, churn risks).
- Product issues (insurance, travel, retail).
"""

ROUTER_SYSTEM_PROMPT = f"""You are a Query Routing Agent.
Your job is to decide if a user's input requires retrieving documents from a specific database or if it can be handled by the LLM's internal knowledge/chat history.

DATASET DESCRIPTION:
{RAG_DOMAIN_DESCRIPTION}

ROUTING LOGIC:
1. **"RAG"**: Choose this if the query asks about agents, customers, tickets, refunds, specific business metrics, or anything related to the dataset description above.
2. **"NO_RAG"**: Choose this if the query is:
   - A question about the current chat history (e.g., "What did I just say?", "Repeat the last answer").
   - A general greeting (e.g., "Hi", "Who are you?").
   - A general knowledge question unrelated to the specific dataset (e.g., "Write python code", "What is 2+2?").
   - A request to clarify the AI's capabilities.

OUTPUT FORMAT:
Return a single JSON object:
{{
    "action": "RAG" or "NO_RAG",
    "reasoning": "Brief explanation of why"
}}
"""

def route_query(query_text: str) -> dict:
    """
    Returns {'action': 'RAG' | 'NO_RAG', 'reasoning': '...'}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query_text}
            ],
            temperature=0,
            response_format={"type": "json_object"} 
        )

        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        else:
            raise ValueError("Empty response from model")

    except Exception as e:
        print(f"Router Error: {e}")
        return {"action": "RAG", "reasoning": "Error fallback"}
