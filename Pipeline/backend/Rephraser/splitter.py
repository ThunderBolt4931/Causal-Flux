import json
from openai import OpenAI
import os
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  

SYSTEM_PROMPT = """You are a “query splitter.”  
When given a multihop user query, you should split it into its constituent sub-queries — but only those parts which are explicit (or almost explicit) in the user’s text. Do *not* invent additional steps or infer implicit reasoning.  

Rules:
1. You may split on visible conjunctions or punctuation like “and”, “then”, “;”, “or” — when they clearly separate sub-questions.  
2. Each sub-query must itself be a causal query (i.e. asking about actions, decisions, causes or plans), or at least intended as a step toward a decision/action.  
3. If a visible part is factual (just asks for facts), *do not* separate it out alone. Instead attach it to the nearest causal sub-query if it's clearly embedded.  
4. Preserve roughly the user’s original wording (you may trim whitespace).  
5. Return a JSON list of objects. Each object:  
   - id: integer, 1-based  
   - text: the subquery string  
   - depends_on: list of integers — for simplicity, make each subquery depend on the immediately previous one (i.e. linear order)  
   - type: "causal" 
6. If you think a query is better answered by not splitting it at all, return the query as it is. 

Output only the JSON."""

def split_query(user_query: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        temperature=0.1
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError("Model did not return valid JSON:\n" + text)