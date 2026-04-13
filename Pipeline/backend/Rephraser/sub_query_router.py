from openai import OpenAI
import json
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_retrieval_pipeline(query: str):
    """
    Replace with your real retrieval pipeline.
    """
    return f"[RETRIEVAL PIPELINE CALLED FOR]: {query}"

def process_query_with_linear_context(last_queries, current_query):
    """
    Decides if we need retrieval based on HISTORY + CURRENT QUERY.
    """
    system_prompt = """
You are an LLM. You get up to 3 previous user queries (linear history).
Your task:

1. Check if the current query can be answered without new documents.
2. If answerable → set "doable": true and produce an "answer".
3. If NOT answerable → set "doable": false. 
   In this case:
       - DO NOT answer the query yourself.
       - Instead, produce a refined retrieval query in "retrieval_query".
4.Only retrieve documents if absolutely necessary

STRICT JSON OUTPUT:
{
  "doable": true/false,
  "answer": "string (may be empty)",
  "retrieval_query": "string"
}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": json.dumps({
             "previous_queries": last_queries,
             "current_query": current_query
         }, indent=2)}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"doable": False, "retrieval_query": current_query}