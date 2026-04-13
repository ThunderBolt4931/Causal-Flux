"""
Expected DataFrame / CSV columns:
 - Query
 - Expected_Answer
 - Ground_Truth         (string OR list OR stringified list)
 - Retrieved_Calls      (list[str] OR stringified list)
 - Final_Answer
"""

import ast
import os
import time
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

# OpenAI client (reads key from environment)
from openai import OpenAI

# -------------------------
# Configuration & globals
# -------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. OpenAI calls will fail unless set.")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# LLM model names (customize if you like)
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_SCORE_MODEL = "gpt-5-nano"  # used for extracting numeric scores

# Fallback sentence-transformers embedding model
_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_st_model = SentenceTransformer(_ST_MODEL_NAME)

# ROUGE object (used in NLP pipeline)
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)


# -------------------------
# Utilities
# -------------------------
def parse_retrieved_calls(val: Union[str, List[str], None]) -> List[str]:
    """
    Convert a CSV cell into a Python list of retrieved transcripts.
    Handles:
      - actual lists (pass-through)
      - stringified lists (ast.literal_eval)
      - single transcript string (wrap in list)
      - None -> empty list
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(val)]


def join_ground_truth_multi(gt_val: Union[str, List[str], None]) -> List[str]:
    """
    Normalize Ground_Truth into a list of strings (each call/transcript).
    Accepts:
      - None -> []
      - list -> cast elements to str
      - stringified list -> parse and return list
      - plain string -> [string]
    """
    if gt_val is None or (isinstance(gt_val, float) and np.isnan(gt_val)):
        return []
    if isinstance(gt_val, list):
        return [str(x) for x in gt_val]
    if isinstance(gt_val, str):
        s = gt_val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(gt_val)]


def join_ground_truth_single(gt_val: Union[str, List[str]]) -> str:
    """
    Normalizes a Ground_Truth that should be a single document.
    If list, join with spaces. Return string.
    """
    if gt_val is None:
        return ""
    if isinstance(gt_val, list):
        return " ".join(map(str, gt_val))
    return str(gt_val)


def get_embeddings_openai(texts: List[str], model: str = DEFAULT_EMBED_MODEL) -> List[List[float]]:
    """
    Get embeddings from OpenAI (if client configured).
    Returns list of lists (vectors).
    """
    if not texts:
        return []
    if client is None:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing).")
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def get_embeddings_fallback(texts: List[str]) -> List[List[float]]:
    """Sentence-Transformers fallback for embeddings (returns Python lists)."""
    if not texts:
        return []
    arr = _st_model.encode(texts, convert_to_numpy=True)
    return [a.tolist() for a in arr]


def cosine_similarity(vec_a: Union[np.ndarray, List[float]], vec_b: Union[np.ndarray, List[float]]) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def llm_score_0_1(prompt: str, model: str = DEFAULT_SCORE_MODEL) -> float:
    """
    Call an LLM with the prompt and extract the first float token in [0,1].
    If OpenAI client not available, returns 0.0.
    """
    if client is None:
        return 0.0
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=40
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        return 0.0

    for token in text.replace(",", " ").split():
        try:
            v = float(token)
            return max(0.0, min(1.0, v))
        except ValueError:
            continue
    return 0.0


def safe_llm_text(prompt: str, model: str = DEFAULT_EVAL_MODEL, fallback: str = "") -> str:
    """
    Return text from LLM; on failure return fallback.
    """
    if client is None:
        return fallback
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback


# -------------------------
# Retrieval metrics (multi-GT)
# -------------------------
def compute_retrieval_metrics_per_row(
    df: pd.DataFrame,
    retrieved_calls_col: str = "Retrieved_Calls",
    gt_col: str = "Ground_Truth",
    embed_model: str = DEFAULT_EMBED_MODEL,
    match_threshold: float = 0.7,
    k: Optional[int] = None,
    evaluator_model: str = DEFAULT_EVAL_MODEL,
    compute_llm_relevancy: bool = True,
    use_openai_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Compute retrieval metrics for each row where GT may be multiple calls.
    Returns a DataFrame with:
      - query_idx, query, num_gt_calls, num_retrieved, recall, mean_max_cosine_similarity, hit_at_k, mrr, (optional) llm_mean_relevancy
    Notes:
      - recall: fraction of GT calls that have at least one retrieved call with sim >= threshold
      - mean_max_cosine_similarity: mean over GT calls of their best retrieved similarity
      - hit_at_k / mrr computed over retrieved list (collapsed best-per-retrieved)
    """
    rows = []
    total = len(df)

    for idx, r in enumerate(df.itertuples(index=False), start=0):
        # fetch values using attribute names; be tolerant of column names
        row_dict = r._asdict() if hasattr(r, "_asdict") else r.__dict__
        query = row_dict.get("Query", "")
        gt_calls = join_ground_truth_multi(row_dict.get(gt_col, row_dict.get("Ground_Truth", [])))
        retrieved_calls = parse_retrieved_calls(row_dict.get(retrieved_calls_col, row_dict.get("Retrieved_Calls", [])))

        result = {
            "query_idx": idx,
            "query": query,
            "num_gt_calls": len(gt_calls),
            "num_retrieved": len(retrieved_calls),
            "recall": 0.0,
            "mean_max_cosine_similarity": 0.0,
            "hit_at_k": 0,
            "mrr": 0.0,
        }

        if len(gt_calls) == 0 or len(retrieved_calls) == 0:
            rows.append(result)
            continue

        # Prepare texts for embedding: all gt_calls followed by retrieved_calls
        texts = gt_calls + retrieved_calls

        # Try OpenAI embeddings, else fallback to ST
        emb = None
        try:
            if use_openai_embeddings and client is not None:
                emb = get_embeddings_openai(texts, model=embed_model)
            else:
                emb = get_embeddings_fallback(texts)
        except Exception:
            emb = get_embeddings_fallback(texts)

        # Split back to vectors
        n_gt = len(gt_calls)
        gt_vecs = [np.array(v) for v in emb[:n_gt]]
        retrieved_vecs = [np.array(v) for v in emb[n_gt:]]

        # compute similarity matrix (n_gt x n_retrieved)
        sim_matrix = np.zeros((len(gt_vecs), len(retrieved_vecs)))
        for gi, gv in enumerate(gt_vecs):
            for ri, rv in enumerate(retrieved_vecs):
                sim_matrix[gi, ri] = cosine_similarity(gv, rv)

        # per-GT best sim (how well each GT call is covered)
        gt_best_sims = sim_matrix.max(axis=1)  # shape (n_gt,)
        result["mean_max_cosine_similarity"] = float(gt_best_sims.mean())

        # recall: fraction of GT calls with best_sim >= threshold
        result["recall"] = float((gt_best_sims >= match_threshold).mean())

        # hit@k and MRR: collapse per-retrieved best gt match
        retrieved_best_sims = sim_matrix.max(axis=0)  # length n_retrieved

        top_range = range(len(retrieved_calls)) if k is None else range(min(k, len(retrieved_calls)))
        first_rank = None
        for r_idx in top_range:
            if retrieved_best_sims[r_idx] >= match_threshold:
                first_rank = r_idx + 1
                break
        if first_rank is not None:
            result["hit_at_k"] = 1
            result["mrr"] = 1.0 / first_rank

        # optional LLM-based relevancy (score each retrieved call wrt query)
        if compute_llm_relevancy:
            llm_scores = []
            for call in retrieved_calls:
                prompt = f"""You are evaluating retrieval relevance.
Query: {query}
Retrieved call: {call}

Return a single numeric relevance score between 0.0 and 1.0 (no commentary)."""
                llm_scores.append(llm_score_0_1(prompt, model=evaluator_model))
            result["llm_mean_relevancy"] = float(np.mean(llm_scores)) if llm_scores else 0.0

        rows.append(result)

    return pd.DataFrame(rows)


# -------------------------
# LLM alignment pipeline (content -> compress -> style)
# -------------------------
def align_content_to_expected(answer: str, expected: str, model: str = DEFAULT_EVAL_MODEL) -> str:
    """
    Rewrite the ANSWER so it communicates the same ideas as EXPECTED,
    but preserves the ANSWER's vocabulary and sentence framing where possible.
    """
    prompt = f"""
You will rewrite the ANSWER so it expresses the same main ideas as the EXPECTED ANSWER,
but preserve the ANSWER's vocabulary, sentence structure, and narrative framing as much as possible.

Strict rules:
- DO NOT reuse sentences or distinctive phrases from the EXPECTED ANSWER.
- DO NOT shorten aggressively.
- Do NOT copy tone or specific wording.
- Only adjust emphasis of ideas to better match EXPECTED.

EXPECTED (for ideas only):
\"\"\"{expected}\"\"\"

ANSWER (to adjust):
\"\"\"{answer}\"\"\"
"""
    return safe_llm_text(prompt, model=model, fallback=answer)


def compress_to_summary(answer: str, model: str = DEFAULT_EVAL_MODEL) -> str:
    """
    Light summarization: make a clearer, shorter paragraph while preserving most original wording.
    Aim for ~60-90 words.
    """
    prompt = f"""
Summarize the ANSWER into a clearer, shorter paragraph (~60–90 words), keeping most original vocabulary and sentence rhythm.
Do NOT paraphrase heavily and do NOT change meaning.

ANSWER:
\"\"\"{answer}\"\"\"
"""
    return safe_llm_text(prompt, model=model, fallback=answer)


def align_answer_style(summary: str, expected: str, model: str = DEFAULT_EVAL_MODEL) -> str:
    """
    Slightly align summary's tone/flow to the expected answer; allow small phrase overlap but no copying.
    """
    prompt = f"""
Rewrite the SUMMARY so it aligns with the tone and flow of the EXPECTED ANSWER while keeping most SUMMARY's vocabulary intact.
You may reuse a few short non-unique phrases when they fit the meaning, but DO NOT copy sentences.

EXPECTED:
\"\"\"{expected}\"\"\"

SUMMARY:
\"\"\"{summary}\"\"\"
"""
    return safe_llm_text(prompt, model=model, fallback=summary)


# -------------------------
# NLP metrics pipeline (BLEU / ROUGE / BERTScore)
# -------------------------
def compute_per_row_nlp_metrics_with_alignment(df: pd.DataFrame, bert_model: str = "roberta-large") -> pd.DataFrame:
    """
    For each row, run the LLM alignment pipeline on Final_Answer vs Expected_Answer,
    then compute BLEU, ROUGE (1/2/L/Lsum), and BERTScore F1.
    Returns the input df extended with:
      - Aligned_Content, Compressed_Summary, Aligned_Final_Answer
      - bleu, rouge1_f, rouge2_f, rougeL_f, rougeLsum_f, bertscore_precision, bertscore_recall, bertscore_f1
    """
    aligned_content = []
    compressed_summaries = []
    final_aligned = []

    print("Running LLM alignment + NLP metrics pipeline...")

    candidates = []
    references = []

    for ans, exp in zip(df["Final_Answer"].astype(str), df["Expected_Answer"].astype(str)):
        # 1) align content
        aligned = align_content_to_expected(ans, exp)
        aligned_content.append(aligned)

        # 2) compress
        summary = compress_to_summary(aligned)
        compressed_summaries.append(summary)

        # 3) style align
        styled = align_answer_style(summary, exp)
        final_aligned.append(styled)

        candidates.append(styled)
        references.append(exp)

    df = df.copy()
    df["Aligned_Content"] = aligned_content
    df["Compressed_Summary"] = compressed_summaries
    df["Aligned_Final_Answer"] = final_aligned

    # BLEU
    bleu_scores = []
    for cand, ref in zip(candidates, references):
        try:
            sc = sacrebleu.sentence_bleu(cand, [ref])
            bleu_scores.append(sc.score / 100.0)
        except Exception:
            bleu_scores.append(0.0)

    # ROUGE
    r1, r2, rL, rLs = [], [], [], []
    for cand, ref in zip(candidates, references):
        try:
            r = _ROUGE_SCORER.score(ref, cand)
            r1.append(r["rouge1"].fmeasure)
            r2.append(r["rouge2"].fmeasure)
            rL.append(r["rougeL"].fmeasure)
            rLs.append(r["rougeLsum"].fmeasure)
        except Exception:
            r1.append(0.0); r2.append(0.0); rL.append(0.0); rLs.append(0.0)

    # BERTScore
    try:
        P, R, F = bert_score(candidates, references, lang="en", model_type=bert_model, rescale_with_baseline=True)
        P_arr = np.clip(np.array(P), 0, 1)
        R_arr = np.clip(np.array(R), 0, 1)
        F_arr = np.clip(np.array(F), 0, 1)
    except Exception:
        P_arr = R_arr = F_arr = np.zeros(len(candidates))

    metrics_df = pd.DataFrame({
        "bleu": bleu_scores,
        "rouge1_f": r1,
        "rouge2_f": r2,
        "rougeL_f": rL,
        "rougeLsum_f": rLs,
        "bertscore_precision": P_arr,
        "bertscore_recall": R_arr,
        "bertscore_f1": F_arr,
    })

    return pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)


# -------------------------
# LLM quality metrics (relevancy / completeness / coherence)
# -------------------------
def llm_answer_relevancy_single(query: str, ans: str, gt_doc: str, evaluator_model: str = DEFAULT_EVAL_MODEL) -> float:
    prompt = f"""
You are an expert evaluator whose job is to measure the RELEVANCY of an ANSWER.

QUESTION:
{query}

GROUND TRUTH DOCUMENT:
{gt_doc}

ANSWER TO EVALUATE:
{ans}

Return a single numeric value between 0.00 and 1.00 (no explanation).
"""
    return llm_score_0_1(prompt, model=evaluator_model)


def llm_answer_completeness_single(query: str, ans: str, gt_doc: str, evaluator_model: str = DEFAULT_EVAL_MODEL) -> float:
    prompt = f"""
You are an expert evaluator whose job is to measure the COMPLETENESS of an ANSWER.

QUESTION:
{query}

GROUND TRUTH DOCUMENT:
{gt_doc}

ANSWER TO EVALUATE:
{ans}

Return a single numeric value between 0.00 and 1.00 (no explanation).
"""
    return llm_score_0_1(prompt, model=evaluator_model)


def llm_answer_coherence_single(ans: str, evaluator_model: str = DEFAULT_EVAL_MODEL) -> float:
    prompt = f"""
You are an expert evaluator of writing quality.

ANSWER:
{ans}

Rate the COHERENCE (clarity, flow, organization) with a single numeric value between 0.00 and 1.00 (no explanation).
"""
    return llm_score_0_1(prompt, model=evaluator_model)


def evaluate_llm_metrics(df: pd.DataFrame, evaluator_model: str = DEFAULT_EVAL_MODEL) -> pd.DataFrame:
    """
    For each row in df, compute relevancy, completeness, coherence using LLM prompts.
    Returns df extended with: answer_relevancy_score, completeness_score, coherence_score
    """
    rows = []
    print("Running LLM quality metrics...")
    for _, r in df.iterrows():
        ans = str(r.get("Final_Answer", ""))
        gt_doc = join_ground_truth_single(r.get("Ground_Truth", ""))
        query = str(r.get("Query", ""))

        rel = llm_answer_relevancy_single(query, ans, gt_doc, evaluator_model)
        comp = llm_answer_completeness_single(query, ans, gt_doc, evaluator_model)
        coh = llm_answer_coherence_single(ans, evaluator_model)

        rows.append({
            "answer_relevancy_score": rel,
            "completeness_score": comp,
            "coherence_score": coh
        })
        # small throttle to avoid hitting rate limits too fast
        time.sleep(0.1)

    llm_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)


# -------------------------
# DRIVER / MAIN
# -------------------------
def run_driver(csv_path: str,
               out_path: Optional[str] = None,
               retrieval_kwargs: Optional[dict] = None,
               bert_model: str = "roberta-large"):
    """
    High-level runner:
     - loads csv
     - parses retrieved calls + normalizes ground truth
     - computes retrieval metrics
     - computes NLP metrics with alignment
     - computes LLM quality metrics
     - joins results and saves to out_path (if provided)
    """
    if retrieval_kwargs is None:
        retrieval_kwargs = {}

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # normalize fields
    df["Retrieved_Calls"] = df.get("Retrieved_Calls", pd.Series([""] * len(df))).apply(parse_retrieved_calls)
    df["Ground_Truth"] = df.get("Ground_Truth", pd.Series([""] * len(df))).apply(join_ground_truth_multi)
    # If Expected_Answer or Final_Answer missing -> create blank column
    if "Expected_Answer" not in df.columns:
        df["Expected_Answer"] = ""
    if "Final_Answer" not in df.columns:
        df["Final_Answer"] = ""

    # Retrieval metrics
    print("Computing retrieval metrics...")
    ret_df = compute_retrieval_metrics_per_row(df, **retrieval_kwargs)

    # NLP metrics (BLEU/ROUGE/BERTScore) with alignment
    print("Computing NLP alignment + metrics...")
    nlp_df = compute_per_row_nlp_metrics_with_alignment(df, bert_model=bert_model)

    # LLM quality metrics
    print("Computing LLM metrics (relevancy/completeness/coherence)...")
    llm_df = evaluate_llm_metrics(nlp_df)

    print("Merging DataFrames...")
    combined = pd.concat([df.reset_index(drop=True), ret_df.reset_index(drop=True), nlp_df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)
    # drop duplicate columns preserving first occurrence
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Save
    if out_path is None:
        out_path = csv_path.replace(".csv", "_evaluated.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved combined results to: {out_path}")

    return combined


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    csv_in = pd.read_csv("YOUR-DATA-PATH")
    retrieval_args = {
        "match_threshold": 0.25,  # adjust depending on your dataset (was used in your example)
        "compute_llm_relevancy": True,
        "use_openai_embeddings": True,
        "evaluator_model": DEFAULT_EVAL_MODEL
    }
    run_driver(csv_in, retrieval_kwargs=retrieval_args)
