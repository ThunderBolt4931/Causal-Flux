from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    print(f"Reranker model '{RERANKER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Failed to load reranker model: {e}")
    reranker = None

def rerank_documents(query: str, documents: List[Document], final_k: int = 5) -> List[Document]:
    """
    Rerank a list of Documents using a CrossEncoder model.
    """
    if not documents or reranker is None:
        return documents[:final_k]
    
    pairs = [[query, doc.page_content] for doc in documents]
    
    scores = reranker.predict(pairs)
    
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:final_k]]
