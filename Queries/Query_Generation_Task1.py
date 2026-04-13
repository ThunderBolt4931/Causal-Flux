import json
import os
import tqdm
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import re
import unicodedata
import random

x = {}
with open('mergedout.json', 'r') as f:
  x = json.load(f)

data = x

all_data = {}        # transcript_id → full json

for file in data:
  tid = file['transcript_id']
  all_data[tid] = file

y = {}
with open('clustered_transcripts.json', 'r') as f:
  y = json.load(f)

clusters = y

client = "<your client model>"

def generate_multihop_questions_with_llm(context_text):
    """
    Sends context to Groq LLM and returns one grounded, generalized, harder multi-hop causal reasoning question.
    """

    prompt = f"""
You are an expert at generating **hard, multi-hop, causal reasoning questions** from call transcripts.

Your Task:
1. Generate exactly ONE question.
2. Extract the ground-truth answer from the call context.
3. Output a strict JSON object:
   - "question"
   - "ground_truth"

GENERALIZED QUESTION SCOPE:
Your question may target:
- emotional reactions (customer or agent)
- escalation or de-escalation events
- misunderstandings or corrections
- conversation outcomes
- changes in tone or behavior
- conflict buildup or resolution

MULTI-HOP REQUIREMENT:
- The question's answer must involve a **2–3 step causal chain**.
- No hallucinations, do not be very specific in the question, keep it a bit generalized.

STRONG VARIATION REQUIREMENT (DO NOT REPEAT THEMES):
Avoid these repetitive phrasing patterns:
- “What caused the customer's frustration”
- “What caused the agent’s frustration”
- “What event triggered…”
- Any close variant of the above.

OUTCOME MAY BE:
- an emotion (frustration, anger, confusion, concern, disappointment, relief, satisfaction)
- a behavioral change
- an escalation/transfer
- a clarification
- a decision
- a conflict or de-escalation

REQUIREMENTS:
- No explanations, keep it concise.
- No speculation.
- Ask question the way a human analyst would ask.
- Generalized, do not make it call specific.

GROUND TRUTH:
- Expected answer to the question.
- Minimal paraphrasing allowed.
- No invented steps.

IMPORTANT:
You are given multiple call transcripts.
You MUST read ALL calls.
Your question MAY involve information across calls.
You must NOT restrict yourself to the first call.
The question must be asked in a way a person would ask if he were to inquire about it for the first time.
Equal number of questions from every combination of [easy, hard] x [factual reasoning/causality, emotional reasoning/causality]


Output format (STRICT):
{{
  "question": "<generated question>",
  "ground_truth": "<explicit multi-step answer from context>"
}}

Context:
{context_text}
"""

    response = client.chat.completions.create(
        model="<insert model name>",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

def generate_singlehop_questions_with_llm(context_text):
    """
    Sends context to Groq LLM and returns one grounded, generalized,
    single-hop causal reasoning question.
    """

    prompt = f"""
You are an expert at generating **single-hop causal reasoning questions** from call transcripts.

----------------------------------------
### TASK
Generate **exactly ONE** question and its **ground-truth answer**.

Output a strict JSON object with:
{{
  "question": "<question>",
  "ground_truth": "<answer>"
}}

----------------------------------------
### QUESTION REQUIREMENTS (Single-Hop)
- Must require **one causal link** (NOT multi-hop).
- Must be **generalized**, NOT call-specific.
- Must NOT repeat common phrasing such as:
  - “What caused the customer’s frustration”
  - “What caused the agent’s frustration”
  - “What event triggered…”
  - Any close variants of these.
- Should target one of these dimensions:
  - emotional reactions (customer/agent)
  - misunderstandings or corrections
  - tone shifts
  - conflict or de-escalation
  - small outcomes/decisions within the call

----------------------------------------
### VARIATION REQUIREMENT
Maintain distribution across:
- easy / hard
- factual / emotional reasoning

IMPORTANT:
Do NOT explicitly label difficulty or category.
Just ensure diversity in the question type.

----------------------------------------
### GROUND-TRUTH REQUIREMENTS
- Answer must come **directly from the context**.
- Single-step reasoning only.
- Absolutely no hallucination.
- Minimal paraphrasing allowed.
- No explanations — only the answer.

----------------------------------------
### MULTI-CALL CONTEXT RULE
You are given **multiple calls**.
- You MUST consider ALL of them.
- Your question MAY reference information that appears across calls.
- Do NOT restrict the question to the first call.

----------------------------------------
### OUTPUT FORMAT (STRICT)
Only return:
{{
  "question": "...",
  "ground_truth": "..."
}}

----------------------------------------
### CONTEXT
{context_text}
"""

    response = client.chat.completions.create(
        model="<insert model name>",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def classify_question(question):
    """
    Classifies a generated question into:
    - Difficulty: easy / hard
    - Hop: single-hop / multi-hop
    - Target: customer / agent
    - Cause_Type: emotion / action
    """

    prompt = f"""
    You are a classifier. Categorize the following question.

    Question:
    {question}

    Provide labels in JSON with EXACT keys:
    difficulty: "easy" or "hard"
    hop: "single-hop" or "multi-hop"
    target: "customer" or "agent"
    cause_type: "emotion" or "action"

    Rules:
    - "easy" = simple, shallow cause/effect
    - "hard" = multi-factor, abstract, deeper reasoning
    - "single-hop" = one causal step
    - "multi-hop" = multiple reasoning steps
    - "customer" = the question concerns customer feelings, actions, experiences
    - "agent" = the question concerns agent decisions or behaviors
    - "emotion" = cause of feelings
    - "action" = cause of behaviors/actions

    Output ONLY JSON.
    """

    response = client.chat.completions.create(
        model="<model name>",   # or gpt-4.1 / gpt-4o / etc
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        # fallback safe defaults
        return {
            "difficulty": "easy",
            "hop": "single-hop",
            "target": "customer",
            "cause_type": "action"
        }

def build_context(tids, all_data):
    return "\n".join(str(all_data[tid]["turns"]) for tid in tids)

def clean_question(q):
    return q.lstrip("0123456789.-• ").strip()

def clean_response(text, remove_emojis=True):
    if text is None:
        return ""

    # Normalize Unicode (fixes weird accents, forms, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width and invisible characters
    invisible_chars = [
        "\u200b", "\u200c", "\u200d",        # zero-width spaces
        "\ufeff",                            # BOM
    ]
    for ch in invisible_chars:
        text = text.replace(ch, "")

    # Remove markdown/code block ticks
    text = re.sub(r"`{1,3}", "", text)

    # Remove common bullet characters
    bullets = r"[•●◦▪▫➤►➡✦—–-]"  # includes long dash/em-dash/en-dash
    text = re.sub(fr"^{bullets}\s*", "", text)

    # Remove emojis (optional)
    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)

    # Strip whitespaces
    text = text.strip()

    return text

def generate_single_call_dataset(
    data,
    sample_count,
    llm_func,
    clean_func,
    classify_func
):
    """
    Generates questions from sampled calls.

    Args:
        data (list): List of call objects (JSON).
        sample_count (int): Number of calls to sample.
        llm_func (callable): Function to generate question JSON (e.g., generate_questions_with_llm).
        clean_func (callable): Cleaning function for text fields.
        classify_func (callable): Function to classify question metadata.

    Returns:
        list[dict]: List of dataset rows.
    """

    sampled_calls = random.sample(data, sample_count)
    rows = []

    for call in tqdm(sampled_calls):

        # --- Domain ---
        domain = call['metadata']['completed_fields'].get('domain', None)

        # --- Intents (both detected + predefined) ---
        intents = []
        for item in call['metadata'].get('identified_interaction_drivers', []):
            intents.append(item.get('keyword'))

        for item in call['metadata'].get('predefined_interaction_drivers', []):
            intents.append(item.get('driver'))

        # --- Transcript ID ---
        transcript_id = call.get('transcript_id')

        # --- Call LLM ---
        llm_response = llm_func(str(call['turns']))
        llm_json = json.loads(llm_response)

        # Clean outputs
        question = clean_func(llm_json['question'])
        ground_truth = clean_func(llm_json['ground_truth'])

        # Classify Q metadata
        meta = classify_func(question)

        # --- Append row ---
        rows.append({
            "Question": question,
            "Domains": [domain],
            "Intents": intents,
            "difficulty": meta['difficulty'],
            "hop": meta["hop"],
            "target": meta['target'],
            "cause_type": meta['cause_type'],
            "t_id": [transcript_id],
            "expected_answer": ground_truth
        })

    return rows

def generate_multi_call_dataset(
    clusters,
    all_data,
    max_clusters,
    llm_func,
    clean_func,
    classify_func
):
    """
    Generate multi-call causal reasoning questions from cluster-level call summaries.

    Args:
        clusters (list): List of cluster objects.
        all_data (dict): Mapping of transcript_id -> call data.
        max_clusters (int): Maximum number of clusters to process.
        llm_func (callable): LLM query function (e.g., generate_questions_with_llm).
        clean_func (callable): Text cleaning function.
        classify_func (callable): Question classifier.

    Returns:
        list[dict]: Rows for dataset.
    """

    rows = []
    count = max_clusters

    for cluster in clusters:

        # Skip non-L2 clusters
        if cluster.get("type") != "L2Cluster":
            continue
        if count <= 0:
            break
        count -= 1

        tids = cluster["member_ids"]

        # --- Build merged context across calls ---
        context = ""
        for tid in tids:
            context += str(all_data[tid]["metadata"]["call_summary"])

        # --- LLM call ---
        try:
            llm_response = llm_func(context)
            parsed = json.loads(llm_response)
        except Exception:
            # skip bad generations
            continue

        question = clean_func(parsed.get("question", ""))
        ground_truth = clean_func(parsed.get("ground_truth", ""))

        # --- classify ---
        meta = classify_func(question)

        # --- collect intents ---
        intents = []
        for tid in tids:
            metadata = all_data[tid]["metadata"]
            # Ensure metadata structure is correct
            if len(metadata) != 5:
                continue
            for d in metadata.get("identified_interaction_drivers", []):
                intents.append(d.get("keyword"))

        # --- collect domains ---
        domains = []
        for tid in tids:
            domains.append(all_data[tid]["metadata"]["completed_fields"]["domain"])

        # --- append row ---
        rows.append({
            "Question": question,
            "Domains": domains,
            "Intents": intents,
            "difficulty": meta["difficulty"],
            "hop": meta["hop"],
            "target": meta["target"],
            "cause_type": meta["cause_type"],
            "t_id": tids,
            "expected_answer": ground_truth
        })

    return rows

# single call — single hop
df = pd.DataFrame(
    generate_single_call_dataset(
        data,
        15,
        generate_singlehop_questions_with_llm,
        clean_response,
        classify_question
    )
)
df.to_csv("single_call_single_hop.csv", index=False)



# single call — multi hop
df = pd.DataFrame(
    generate_single_call_dataset(
        data,
        5,
        generate_multihop_questions_with_llm,
        clean_response,
        classify_question
    )
)
df.to_csv("single_call_multi_hop.csv", index=False)



# multi call — single hop
df = pd.DataFrame(
    generate_multi_call_dataset(
        clusters,
        all_data,
        130,
        generate_singlehop_questions_with_llm,
        clean_response,
        classify_question
    )
)
df.to_csv("multi_call_single_hop.csv", index=False)



# multi call — multi hop
df = pd.DataFrame(
    generate_multi_call_dataset(
        clusters,
        all_data,
        50,
        generate_multihop_questions_with_llm,
        clean_response,
        classify_question
    )
)
df.to_csv("multi_call_multi_hop.csv", index=False)
