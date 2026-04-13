#!/usr/bin/env python3
"""
Compact refactor of the clustering + labeling pipeline.
Preserves original algorithm: embeddings -> L2 agglomerative -> label L2 -> compose L2 embeddings -> L1 agglomerative -> label L1 -> save outputs.
"""

import os
import json
import time
import random
import asyncio
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from groq import AsyncGroq, RateLimitError, InternalServerError, APIConnectionError

# --------- Config ----------
INPUT_FILE = "Corpus\corpus.json"
OUT_DIR = "Clusters/"
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 1000
NUM_L2 = 200
NUM_L1 = 20
MMR_TOP_K = 20
MMR_LAMBDA = 0.6
WEIGHT = {"centroid": 1.0, "name": 0.5, "desc": 0.5}
API_KEYS = [] # enter youe groq api keys in this list. They will be used asynchronously while staying within rate limits
LLM_CONCURRENCY = 7
LLM_BATCH = 30

# --------- Small helpers ----------
def load_data(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def embed_texts(texts: List[str], model: str = EMBED_MODEL, batch: int = EMBED_BATCH) -> np.ndarray:
    if not texts:
        return np.array([])
    client = OpenAI()
    embs = []
    for i in tqdm(range(0, len(texts), batch), desc=f"Embedding {len(texts)}"):
        batch_texts = texts[i:i+batch]
        resp = client.embeddings.create(input=batch_texts, model=model)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs)

def mmr_pick(doc_emb: np.ndarray, centroid: np.ndarray, top_k: int = 5, lam: float = 0.6) -> List[int]:
    if len(doc_emb) == 0:
        return []
    centroid = centroid.reshape(1, -1)
    sim_cent = cosine_similarity(doc_emb, centroid).flatten()
    selected, candidates = [], list(range(len(doc_emb)))
    for _ in range(min(top_k, len(doc_emb))):
        best_idx, best_score = -1, -1e9
        for c in candidates:
            rel = float(sim_cent[c])
            if not selected:
                div_pen = 0.0
            else:
                sim_sel = cosine_similarity(doc_emb[c].reshape(1, -1), doc_emb[selected]).flatten()
                div_pen = float(sim_sel.max())
            score = lam * rel - (1-lam) * div_pen
            if score > best_score:
                best_score, best_idx = score, c
        if best_idx == -1:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

# --------- LLM labeling (compact) ----------
async def _llm_label_once(client: AsyncGroq, prompt: str, cid: int, level: str, field: str) -> Tuple[str,str]:
    try:
        resp = await client.chat.completions.create(
            messages=[
                {"role":"system","content":"You are a helpful assistant that outputs JSON."},
                {"role":"user","content":prompt},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.3,
            response_format={"type":"json_object"}
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("name", f"{level} {cid}"), parsed.get("description","General grouping")
    except Exception:
        raise

async def llm_label_with_retries(prompt: str, cid: int, level: str, field: str, tries: int = 6):
    delay = 2.0
    for attempt in range(tries):
        api_key = random.choice(API_KEYS)
        client = AsyncGroq(api_key=api_key)
        try:
            name, desc = await _llm_label_once(client, prompt, cid, level, field)
            await client.close()
            return name, desc
        except (RateLimitError, InternalServerError, APIConnectionError):
            await client.close()
            await asyncio.sleep(delay * (2**attempt) + random.random())
        except Exception:
            await client.close()
            break
    return f"{field} {level} {cid}", "General grouping"

async def label_batch(tasks: List[Dict], concurrency: int = LLM_CONCURRENCY) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)
    async def run_task(t):
        async with sem:
            await asyncio.sleep(random.random()*0.5)
            n,d = await llm_label_with_retries(t["prompt"], t["cid"], t["level"], t["field"])
            t["name"], t["description"] = n, d
            return t
    out = []
    for i in range(0, len(tasks), LLM_BATCH):
        batch = tasks[i:i+LLM_BATCH]
        out.extend(await asyncio.gather(*(run_task(t) for t in batch)))
    return out

# --------- Core: cluster & label level ----------
def agglomerative_labels(emb_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(emb_matrix) == 0:
        return np.array([], dtype=int)
    n_clusters = min(n_clusters, len(emb_matrix))
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(emb_matrix)

async def cluster_and_label_level(docs: List[str],
                                  doc_ids: List[str],
                                  doc_emb: np.ndarray,
                                  level_name: str,
                                  target_clusters: int) -> Tuple[List[dict], Dict]:
    # L2-like step: cluster docs -> make prompts -> label L2 -> return cluster objects + transcript map
    labels = agglomerative_labels(doc_emb, target_clusters)
    if labels.size == 0:
        return [], {}
    cluster_ids = np.unique(labels)
    tasks = []
    cluster_meta = {}
    for cid in cluster_ids:
        idxs = np.where(labels == cid)[0]
        cluster_embs = doc_emb[idxs]
        centroid = cluster_embs.mean(axis=0)
        mmr_idxs = mmr_pick(cluster_embs, centroid, top_k=MMR_TOP_K, lam=MMR_LAMBDA)
        rep_texts = [docs[idxs[i]] for i in mmr_idxs]
        prompt = (
            f"You are analyzing a cluster of {level_name} for customer-service text.\n"
            f"Representative samples:\n" + "\n---\n".join(rep_texts) + "\n\n"
            "Return JSON: {\"name\":\"...\",\"description\":\"...\"}"
        )
        tasks.append({"cid": int(cid), "indices": idxs.tolist(), "centroid": centroid, "prompt": prompt, "level": level_name, "field": ""})
    labeled = await label_batch(tasks)
    # compute name/desc embs and combined embedding for each cluster
    names = [t["name"] for t in labeled]
    descs = [t["description"] for t in labeled]
    name_embs = embed_texts(names) if names else np.array([])
    desc_embs = embed_texts(descs) if descs else np.array([])
    clusters_out = []
    for i, t in enumerate(labeled):
        c_emb = t["centroid"]
        if name_embs.size:
            n_emb = name_embs[i]
            d_emb = desc_embs[i]
            combined = (WEIGHT["centroid"]*c_emb + WEIGHT["name"]*n_emb + WEIGHT["desc"]*d_emb)
            combined = combined / np.linalg.norm(combined)
        else:
            combined = c_emb / np.linalg.norm(c_emb)
        clusters_out.append({
            "cluster_id": int(t["cid"]),
            "name": t["name"],
            "description": t["description"],
            "embedding": combined,
            "doc_indices": t["indices"]
        })
        cluster_meta[int(t["cid"])] = {"name": t["name"], "description": t["description"], "embedding": combined}
    # transcript map
    transcript_map = {}
    for i, tid in enumerate(doc_ids):
        transcript_map[tid] = {
            "embeds": doc_emb[i].tolist(),
            "parent_cluster": f"{level_name}_{int(labels[i])}"
        }
    return clusters_out, transcript_map

# --------- Full field processing ----------
async def process_field(records: List[dict], field: str):
    # extract texts & ids
    texts, ids = [], []
    for r in records:
        tid = r.get("transcript_id")
        if not tid:
            continue
        if field == "keywords":
            kws = r.get("metadata", {}).get("identified_interaction_drivers", [])
            if kws:
                content = ", ".join([k.get("keyword","") for k in kws if k.get("keyword")])
            else:
                content = ""
        else:
            content = r.get(field, "") or r.get("metadata", {}).get(field, "") or ""
        if content:
            texts.append(content)
            ids.append(tid)
    if not texts:
        print(f"No data for {field}, skipping.")
        return
    # embed docs
    doc_emb = embed_texts(texts)
    # L2: cluster docs and label
    l2_objs, transcript_map = await cluster_and_label_level(texts, ids, doc_emb, "L2", NUM_L2)
    if not l2_objs:
        print(f"No L2 objects for {field}")
        return
    # prepare L2 embeddings for L1 clustering
    l2_embs = np.array([o["embedding"] for o in l2_objs])
    # L1: cluster L2 groups and label (treat L2 names & descriptions as inputs inside cluster_and_label_level)
    # Build pseudo-docs for L1 prompt (use "name\n---\ndesc" as doc)
    l2_texts_for_l1 = [f"{o['name']}\n---\n{o['description']}" for o in l2_objs]
    l2_ids_for_l1 = [str(o["cluster_id"]) for o in l2_objs]
    l1_objs, _ = await cluster_and_label_level(l2_texts_for_l1, l2_ids_for_l1, l2_embs, "L1", NUM_L1)
    # Combine outputs into final structures
    clusters_output = []
    for l1 in l1_objs:
        clusters_output.append({
            "type": "L1Cluster",
            "field": field,
            "id": f"{field}_L1_{l1['cluster_id']}",
            "name": l1["name"],
            "description": l1["description"],
            "embedding": l1["embedding"].tolist(),
            "child_count": 0  # unknown here without mapping; optional
        })
    for l2 in l2_objs:
        clusters_output.append({
            "type": "L2Cluster",
            "field": field,
            "id": f"{field}_L2_{l2['cluster_id']}",
            "name": l2["name"],
            "description": l2["description"],
            "embedding": l2["embedding"].tolist(),
            "parent_id": None,  # can map parent by matching embeddings if desired
            "member_ids": [ids[i] for i in l2["doc_indices"]]
        })
    # save
    save_json(clusters_output, os.path.join(OUT_DIR, f"clusters_{field}.json"))
    save_json({field: transcript_map}, os.path.join(OUT_DIR, f"embeddings_{field}.json"))
    print(f"Saved {len(clusters_output)} clusters and {len(transcript_map)} transcript embeddings for {field}.")

# --------- Orchestration ----------
def run():
    start = time.time()
    if not os.path.exists(INPUT_FILE):
        print("Input missing:", INPUT_FILE); return
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_data(INPUT_FILE)
    print("Loaded", len(records), "records.")
    fields = ["keywords", "reason_for_call"]
    async def runner():
        for f in fields:
            await process_field(records, f)
    asyncio.run(runner())
    print("Done in {:.2f}s".format(time.time()-start))

if __name__ == "__main__":
    run()
