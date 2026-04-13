import os
import json
import numpy as np
import time
import asyncio
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from groq import AsyncGroq, RateLimitError, InternalServerError, APIConnectionError
from tqdm import tqdm

# Defined API Keys
API_KEYS = [
    "gsk_",
    "gsk_",...
]

def run_clustering_pipeline():

    INPUT_FILE = "time_and_domain_fixed.json"
    OUTPUT_DIR = "clusters/"

    # Clustering Parameters
    NUM_L2_CLUSTERS = 200
    NUM_L1_CLUSTERS = 20
    W_CENTROID = 1.0
    W_NAME = 0.5
    W_DESC = 0.5
    MMR_TOP_K = 20
    MMR_LAMBDA = 0.6

    print("Initializing OpenAI client...", flush=True)
    openai_client = OpenAI()

    def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=1000):
        if not texts:
            return np.array([])
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {len(texts)} items"):
            batch = texts[i:i + batch_size]
            response = openai_client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def extract_field_data(data, field_type):
        extracted_texts = []
        ids = []
        for item in data:
            tid = item.get('transcript_id')
            content = ""
            if field_type == "summary":
                content = item.get('metadata', {}).get('call_summary', "")
            elif field_type == "outcome":
                content = item.get('metadata', {}).get('outcome', "")
            elif field_type == "keywords":
                keywords_list = item.get('metadata', {}).get('identified_interaction_drivers', {})
                if keywords_list:
                    content = ", ".join([l['keyword'] for l in keywords_list])
            elif field_type == "reason_for_call":
                content = item.get('reason_for_call', "")
            
            if content and tid:
                extracted_texts.append(content)
                ids.append(tid)
        return extracted_texts, ids

    def mmr_selection(doc_embeddings, centroid, top_k=5, diversity_lambda=0.6):
        selected_indices = []
        candidate_indices = list(range(len(doc_embeddings)))
        
        centroid = centroid.reshape(1, -1)
        sim_to_centroid = cosine_similarity(doc_embeddings, centroid).flatten()
        
        for _ in range(min(top_k, len(doc_embeddings))):
            best_score = -np.inf
            best_idx = -1
            
            for idx in candidate_indices:
                relevance = sim_to_centroid[idx]
                
                if not selected_indices:
                    diversity_penalty = 0
                else:
                    selected_embs = doc_embeddings[selected_indices]
                    candidate_emb = doc_embeddings[idx].reshape(1, -1)
                    sim_to_selected = cosine_similarity(candidate_emb, selected_embs).flatten()
                    diversity_penalty = np.max(sim_to_selected)
                
                mmr_score = (diversity_lambda * relevance) - ((1 - diversity_lambda) * diversity_penalty)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        return selected_indices

    async def generate_label_async(client, prompt, cluster_id, field_type, level="L2"):
        max_retries = 6
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model="openai/gpt-oss-120b", 
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                content_str = chat_completion.choices[0].message.content
                content = json.loads(content_str)
                return content.get("name", f"{level} Group {cluster_id}"), content.get("description", "General grouping")
            
            except RateLimitError:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 2.0)
                await asyncio.sleep(wait_time)
            except (InternalServerError, APIConnectionError):
                wait_time = 5.0
                await asyncio.sleep(wait_time)
            except Exception:
                break 
        
        return f"{field_type} {level} Group {cluster_id}", "General grouping"

    async def process_field_async(data, field_type):
        print(f"\n--- Processing Field: {field_type} ---", flush=True)
        all_texts, all_ids = extract_field_data(data, field_type)
        if not all_texts:
            print(f"No text found for {field_type}", flush=True)
            return [], {}

        doc_embeddings = get_openai_embeddings(all_texts)


        print(f"Running L2 Clustering on {len(doc_embeddings)} items...", flush=True)
        l2_model = AgglomerativeClustering(n_clusters=min(NUM_L2_CLUSTERS, len(all_texts)))
        l2_labels = l2_model.fit_predict(doc_embeddings)

        unique_l2_labels = np.unique(l2_labels)
        l2_tasks = []
        
        for cid in unique_l2_labels:
            indices = np.where(l2_labels == cid)[0]
            cluster_embeddings = doc_embeddings[indices]
            cluster_texts = [all_texts[i] for i in indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            mmr_local_indices = mmr_selection(
                cluster_embeddings, 
                centroid, 
                top_k=MMR_TOP_K, 
                diversity_lambda=MMR_LAMBDA
            )
            rep_docs = [cluster_texts[i] for i in mmr_local_indices]
            combined_text = "\n---\n".join(rep_docs)
            
            prompt = f"""You are analyzing a cluster of {field_type} from customer service data.
            Here are {len(rep_docs)} representative samples:
            {combined_text}
            
            Task:
            1. Provide a short, specific Cluster Name (2-4 words).
            2. Provide a 1-sentence Description of the common theme.
            
            Return valid JSON only format: {{"name": "...", "description": "..."}}
            """
            
            l2_tasks.append({
                "cid": cid,
                "indices": indices,
                "centroid": centroid,
                "prompt": prompt
            })

        sem = asyncio.Semaphore(7) 
        
        async def bounded_generate(task_info, level="L2"):
            async with sem:
                await asyncio.sleep(random.uniform(0.1, 1.0))
                api_key = random.choice(API_KEYS)
                client = AsyncGroq(api_key=api_key)
                try:
                    name, desc = await generate_label_async(client, task_info["prompt"], task_info["cid"], field_type, level)
                finally:
                    await client.close()
                return {**task_info, "name": name, "description": desc}

        l2_results = []
        batch_size = 30
        print(f"Generating labels for {len(l2_tasks)} L2 clusters...", flush=True)
        for i in tqdm(range(0, len(l2_tasks), batch_size), desc="L2 LLM Requests"):
            batch = l2_tasks[i:i+batch_size]
            results = await asyncio.gather(*(bounded_generate(t, "L2") for t in batch))
            l2_results.extend(results)

        print("Embedding generated L2 names and descriptions...", flush=True)
        l2_names = [r["name"] for r in l2_results]
        l2_descs = [r["description"] for r in l2_results]
        l2_name_embs = get_openai_embeddings(l2_names)
        l2_desc_embs = get_openai_embeddings(l2_descs)
        
        l2_clusters_data = []
        for i, res in enumerate(l2_results):
            centroid = res["centroid"]
            name_emb = l2_name_embs[i]
            desc_emb = l2_desc_embs[i]
            
            mv_embedding = (W_CENTROID * centroid) + (W_NAME * name_emb) + (W_DESC * desc_emb)
            mv_embedding = mv_embedding / np.linalg.norm(mv_embedding)
            
            l2_clusters_data.append({
                "cluster_id": int(res["cid"]),
                "name": res["name"],
                "description": res["description"],
                "embedding": mv_embedding,
                "doc_indices": res["indices"].tolist() 
            })

        print("Running L1 Clustering...", flush=True)
        l2_mv_embeddings = np.array([c['embedding'] for c in l2_clusters_data])
        
        l1_model = AgglomerativeClustering(n_clusters=min(NUM_L1_CLUSTERS, len(l2_mv_embeddings)))
        l1_labels = l1_model.fit_predict(l2_mv_embeddings) 

        unique_l1_labels = np.unique(l1_labels)
        l1_tasks = []

        for l1_id in unique_l1_labels:
            l2_indices_in_group = np.where(l1_labels == l1_id)[0]
            group_l2_embeddings = l2_mv_embeddings[l2_indices_in_group]
            l1_centroid = np.mean(group_l2_embeddings, axis=0)
            
            mmr_selection_indices = mmr_selection(
                group_l2_embeddings,
                l1_centroid,
                top_k=MMR_TOP_K,
                diversity_lambda=MMR_LAMBDA
            )
            
            rep_l2_clusters_info = []
            for idx in mmr_selection_indices:
                l2_idx_global = l2_indices_in_group[idx]
                l2_obj = l2_clusters_data[l2_idx_global]
                rep_l2_clusters_info.append(f"Sub-Group Name: {l2_obj['name']}\nDescription: {l2_obj['description']}")
            
            combined_l1_text = "\n---\n".join(rep_l2_clusters_info)

            prompt = f"""You are analyzing a high-level category (Super-Cluster) of {field_type} derived from customer service data.
            Here are {len(rep_l2_clusters_info)} representative sub-groups within this category:
            {combined_l1_text}
            
            Task:
            1. Provide a short, specific Category Name (2-4 words).
            2. Provide a 1-sentence Description of the overarching theme connecting these sub-groups.
            
            Return valid JSON only format: {{"name": "...", "description": "..."}}
            """
            
            l1_tasks.append({
                "cid": int(l1_id),
                "centroid": l1_centroid,
                "prompt": prompt,
                "child_l2_indices": l2_indices_in_group.tolist()
            })

        l1_results = []
        print(f"Generating labels for {len(l1_tasks)} L1 clusters...", flush=True)
        for i in tqdm(range(0, len(l1_tasks), batch_size), desc="L1 LLM Requests"):
            batch = l1_tasks[i:i+batch_size]
            results = await asyncio.gather(*(bounded_generate(t, "L1") for t in batch))
            l1_results.extend(results)

        print("Embedding generated L1 names and descriptions...", flush=True)
        l1_names = [r["name"] for r in l1_results]
        l1_descs = [r["description"] for r in l1_results]
        l1_name_embs = get_openai_embeddings(l1_names)
        l1_desc_embs = get_openai_embeddings(l1_descs)

        l1_clusters_final = {}
        cluster_output_objects = []

        for i, res in enumerate(l1_results):
            l1_id = res["cid"]
            centroid = res["centroid"]
            name_emb = l1_name_embs[i]
            desc_emb = l1_desc_embs[i]
            
            mv_embedding = (W_CENTROID * centroid) + (W_NAME * name_emb) + (W_DESC * desc_emb)
            mv_embedding = mv_embedding / np.linalg.norm(mv_embedding)
            
            l1_clusters_final[l1_id] = {
                "name": res["name"],
                "description": res["description"],
                "embedding": mv_embedding
            }
            
            cluster_output_objects.append({
                "type": "L1Cluster",
                "field": field_type,
                "id": f"{field_type}_L1_{l1_id}",
                "name": res["name"],
                "description": res["description"],
                "embedding": mv_embedding.tolist(),
                "child_count": len(res["child_l2_indices"])
            })

        l2_id_to_l1_id = {}
        
        for idx_in_list, l2_data in enumerate(l2_clusters_data):
            l1_label_int = l1_labels[idx_in_list]
            l2_id_int = l2_data['cluster_id']
            l2_id_to_l1_id[l2_id_int] = l1_label_int

            l1_obj = l1_clusters_final[l1_label_int]
            transcript_ids_in_cluster = [all_ids[x] for x in l2_data['doc_indices']]
            
            cluster_output_objects.append({
                "type": "L2Cluster",
                "field": field_type,
                "id": f"{field_type}_L2_{l2_id_int}",
                "name": l2_data['name'],
                "description": l2_data['description'],
                "embedding": l2_data['embedding'].tolist(),
                "parent_id": f"{field_type}_L1_{l1_label_int}",
                "member_ids": transcript_ids_in_cluster
            })

        print("Generating transcript embedding map...", flush=True)
        transcript_map = {}
        
        for i, tid in enumerate(all_ids):
            raw_emb = doc_embeddings[i]
            l2_cluster_id = l2_labels[i]
            l1_cluster_id = l2_id_to_l1_id.get(l2_cluster_id)
            
            transcript_map[tid] = {
                "embeds": raw_emb.tolist(),
                "parent_l2_cluster": f"{field_type}_L2_{l2_cluster_id}",
                "parent_l1_cluster": f"{field_type}_L1_{l1_cluster_id}"
            }

        return cluster_output_objects, transcript_map

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.", flush=True)
        return

    print(f"Loading data from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        FULL_DATA = json.load(f)
    print(f"Loaded {len(FULL_DATA)} records.", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_fields = ["keywords", "reason_for_call"]

    async def main_loop():
        for field in target_fields:
            cluster_data, transcript_data = await process_field_async(FULL_DATA, field)
            
            if not cluster_data:
                print(f"Skipping save for {field} (No data generated).", flush=True)
                continue

            cluster_filename = os.path.join(OUTPUT_DIR, f"clusters_{field}.json")
            print(f"Saving clusters for {field} to {cluster_filename}...", flush=True)
            with open(cluster_filename, 'w', encoding='utf-8') as f:
                json.dump(cluster_data, f, indent=2)
            embed_filename = os.path.join(OUTPUT_DIR, f"embeddings_{field}.json")
            print(f"Saving embeddings for {field} to {embed_filename}...", flush=True)
            with open(embed_filename, 'w', encoding='utf-8') as f:
                json.dump({field: transcript_data}, f, indent=2)
                
            print(f"Completed processing for field: {field}\n", flush=True)

    asyncio.run(main_loop())
