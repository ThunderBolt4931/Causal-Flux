import json
import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
FIELDS = ['summary', 'reason_for_call', 'keywords', 'outcome']

def mmr_select(candidate_embs: np.ndarray, candidate_scores: np.ndarray, k: int, lambda_mmr: float = 0.7) -> List[int]:
    """Selects indices using Maximal Marginal Relevance."""
    if len(candidate_scores) == 0:
        return []

    # Sort available indices
    idxs = list(range(len(candidate_scores)))
    selected = []

    # 1. Pick best relevance first
    first = int(np.argmax(candidate_scores))
    selected.append(first)
    idxs.remove(first)

    # 2. Precompute similarity matrix if needed
    sims = candidate_embs @ candidate_embs.T if len(candidate_embs) > 0 else np.zeros((0, 0))

    # 3. Iteratively pick next best MMR score
    while len(selected) < min(k, len(candidate_scores)):
        mmr_scores = []
        for i in idxs:
            relevance = candidate_scores[i]
            # Max similarity to any already selected item
            max_sim = max(sims[i, j] for j in selected) if selected else 0.0
            score = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim
            mmr_scores.append((score, i))
        
        # Pick highest score
        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected.append(chosen)
        idxs.remove(chosen)

    return selected

class HierarchicalRetriever:
    def __init__(self, clustered_json_path: str, docs_json_path: Optional[str] = None):
        self.client = OpenAI()

        # Load Clusters
        with open(clustered_json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        # Handle dict vs list format
        if isinstance(data, dict):
            self.clusters = data.get("clusters", [])
            embedded_docs = data.get("documents", [])
        else:
            self.clusters = data
            embedded_docs = []

        # Load Documents
        self.docs_map = {}
        source_docs = []
        
        if docs_json_path:
            with open(docs_json_path, 'r', encoding='utf8') as f:
                source_docs = json.load(f)
        elif embedded_docs:
            source_docs = embedded_docs

        # Build Document Lookup
        for d in source_docs:
            if d and 'id' in d:
                self.docs_map[d['id']] = d
        
        # Fallback: Create dummy entries for IDs found in clusters but not in doc list
        for c in self.clusters:
            if c.get('type') == 'L2Cluster':
                for mid in c.get('member_ids', []):
                    if mid not in self.docs_map:
                        self.docs_map[mid] = {'id': mid}

        # Partition Clusters
        self.l1_clusters = [c for c in self.clusters if c.get('type') == 'L1Cluster']
        self.l2_clusters = [c for c in self.clusters if c.get('type') == 'L2Cluster']

    def _embed(self, text: str) -> np.ndarray:
        """Embeds query and normalizes."""
        resp = self.client.embeddings.create(input=[text], model=MODEL_NAME)
        e = np.array(resp.data[0].embedding)
        norm = np.linalg.norm(e)
        return e if norm == 0 else e / norm

    def _process_candidates(self, candidates: List[Dict], query_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts embeddings, normalizes them, and calculates cosine similarity."""
        if not candidates:
            return np.array([]), np.array([])
            
        # Stack and Normalize (adding epsilon for stability)
        raw_embs = np.stack([np.array(c.get('embedding', np.zeros(query_emb.shape)), float) for c in candidates])
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-12
        norm_embs = raw_embs / norms
        
        # Calculate Scores
        scores = cosine_similarity(query_emb[None, :], norm_embs)[0]
        return norm_embs, scores

    def retrieve(self, query: str, field: Optional[str] = None, top_k_l1: int = 3, 
                 top_k_l2_per_l1: int = 4, min_similarity: Optional[float] = None) -> Dict[str, Any]:
        
        LAMBDA_MMR = 0.7
        q_emb = self._embed(query)
        
        # --- PHASE 1: L1 Retrieval (MMR) ---
        l1_candidates = [c for c in self.l1_clusters if field is None or c.get('field') == field]
        l1_embs, l1_scores = self._process_candidates(l1_candidates, q_emb)
        
        sel_l1_indices = mmr_select(l1_embs, l1_scores, k=top_k_l1, lambda_mmr=LAMBDA_MMR)
        selected_l1 = [l1_candidates[i] for i in sel_l1_indices]
        sel_l1_ids = {c['id'] for c in selected_l1}

        # --- PHASE 2: L2 Retrieval (Nested MMR) ---
        selected_l2 = []
        selected_l2_scores = []

        # Optimization: Map ParentID -> List[L2 Clusters]
        l2_candidates = [c for c in self.l2_clusters if field is None or c.get('field') == field]
        l1_to_l2_map: Dict[str, List[Dict]] = {lid: [] for lid in sel_l1_ids}
        for l2 in l2_candidates:
            pid = l2.get('parent_id')
            if pid in l1_to_l2_map:
                l1_to_l2_map[pid].append(l2)

        # Process L2s for each selected L1
        for l1_id in sel_l1_ids:
            children = l1_to_l2_map[l1_id]
            if not children: 
                continue

            child_embs, child_scores = self._process_candidates(children, q_emb)

            # Apply Min Similarity Filter
            valid_indices = list(range(len(children)))
            if min_similarity is not None:
                valid_indices = [i for i in valid_indices if child_scores[i] >= min_similarity]

            if not valid_indices:
                continue

            # Subset for MMR
            filtered_embs = child_embs[valid_indices]
            filtered_scores = child_scores[valid_indices]

            # Run MMR on children
            k_local = min(top_k_l2_per_l1, len(valid_indices))
            local_sel_idx = mmr_select(filtered_embs, filtered_scores, k=k_local, lambda_mmr=LAMBDA_MMR)

            for idx in local_sel_idx:
                original_idx = valid_indices[idx]
                selected_l2.append(children[original_idx])
                selected_l2_scores.append(float(child_scores[original_idx]))

        # --- PHASE 3: Document ID Extraction ---
        final_doc_ids = []
        seen_doc_ids = set()

        for l2 in selected_l2:
            for mid in l2.get('member_ids', []):
                if mid not in seen_doc_ids and mid in self.docs_map:
                    seen_doc_ids.add(mid)
                    final_doc_ids.append(mid)
        
        print(len(final_doc_ids))

        return {
            "query": query,
            "top_l1_ids_considered": list(sel_l1_ids),
            "selected_l2": [
                {"id": c.get("id"), "score": s, "parent_id": c.get("parent_id")}
                for c, s in zip(selected_l2, selected_l2_scores)
            ],
            "documents": final_doc_ids
        }

if __name__ == "__main__":

    CLUSTERED_FILE = "Clusters\clustered_transcripts.json"

    retriever = HierarchicalRetriever(CLUSTERED_FILE)
    test_query = "How does agent empathizing with consumer in their inconvenience prevent churn"

    all_results = {}
    for fld in FIELDS:
        all_results[fld] = retriever.retrieve(
            test_query, field=fld, top_k_l1=3, top_k_l2_per_l1=5, min_similarity=0.25
        )

    os.makedirs("retr", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("retr", f"retrieval_{ts}.json")
    
    with open(out_path, "w", encoding="utf8") as f:
        json.dump({"query": test_query, "results_by_field": all_results}, f, indent=2)

    print(f"Wrote results to {out_path}")