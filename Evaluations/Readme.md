# RAG Evaluation Module

![Metrics](https://img.shields.io/badge/Metrics-BLEU_ROUGE_BERTScore-blue)
![OpenAI](https://img.shields.io/badge/LLM_Eval-OpenAI-green)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

## Overview

This module provides a comprehensive evaluation suite for the RAG pipeline, measuring both **retrieval quality** and **generation quality** using:

- **Hard Metrics**: BLEU, ROUGE, BERTScore
- **LLM-based Metrics**: Relevancy, Completeness, Coherence
- **Retrieval Metrics**: Recall, MRR, Hit@K, Cosine Similarity

---

## Files

| File | Description |
|------|-------------|
| `Evaluations.py` | Main evaluation pipeline with all metrics |

---

## Metrics Overview

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Recall** | Fraction of ground truth calls with similarity >= threshold |
| **MRR** | Mean Reciprocal Rank of first relevant result |
| **Hit@K** | Whether any result in top-K is relevant |
| **Mean Max Cosine Sim** | Average best similarity to each ground truth |

### Generation Metrics (NLP)

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram precision (SacreBLEU) |
| **ROUGE-1** | Unigram overlap |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |
| **BERTScore** | Semantic similarity using RoBERTa |

### LLM-based Metrics

| Metric | Prompt Focus |
|--------|--------------|
| **Relevancy** | Does the answer address the question? |
| **Completeness** | Does it cover all aspects of ground truth? |
| **Coherence** | Is it clear, well-organized, and readable? |

---

## Configuration

```python
# OpenAI API Key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
DEFAULT_EVAL_MODEL = "gpt-4.1-mini"      # For text evaluation
DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # For similarity
DEFAULT_SCORE_MODEL = "gpt-5-nano"       # For numeric scoring

# Fallback embedding model
_ST_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval threshold
match_threshold = 0.7  # Cosine similarity threshold
```

---

## Input Format

Prepare a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `Query` | The user's question |
| `Expected_Answer` | Ground truth answer |
| `Ground_Truth` | Source transcript IDs (list or stringified list) |
| `Retrieved_Calls` | Retrieved transcript IDs from RAG |
| `Final_Answer` | Generated answer from pipeline |

### Example CSV
```csv
Query,Expected_Answer,Ground_Truth,Retrieved_Calls,Final_Answer
"Why did the customer get angry?","The agent was rude","['T001', 'T002']","['T001', 'T003', 'T005']","The customer became frustrated because..."
```

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
```

### Run Evaluation
```python
from Evaluations import run_driver

# Run full evaluation pipeline
results_df = run_driver(
    csv_path="your_test_data.csv",
    out_path="evaluation_results.csv",
    retrieval_kwargs={
        "match_threshold": 0.25,
        "compute_llm_relevancy": True,
        "use_openai_embeddings": True
    },
    bert_model="roberta-large"
)
```

### Command Line
```bash
python Evaluations.py
```

---

## Output Columns

The output CSV includes all input columns plus:

### Retrieval Metrics
- `recall`, `mrr`, `hit_at_k`, `mean_max_cosine_similarity`
- `llm_mean_relevancy` (if enabled)

### NLP Alignment
- `Aligned_Content`, `Compressed_Summary`, `Aligned_Final_Answer`

### NLP Scores
- `bleu`, `rouge1_f`, `rouge2_f`, `rougeL_f`, `rougeLsum_f`
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1`

### LLM Scores
- `answer_relevancy_score`, `completeness_score`, `coherence_score`

---

## Evaluation Pipeline

```
1. Load CSV
      ↓
2. Parse Retrieved Calls & Ground Truth
      ↓
3. Compute Retrieval Metrics
   ├── Embed all texts (OpenAI or SentenceTransformers)
   ├── Build similarity matrix
   ├── Calculate recall, MRR, Hit@K
   └── Optional: LLM relevancy scoring
      ↓
4. LLM Alignment Pipeline
   ├── Align content to expected answer
   ├── Compress to summary
   └── Style alignment
      ↓
5. Compute NLP Metrics
   ├── BLEU (SacreBLEU)
   ├── ROUGE (1, 2, L, Lsum)
   └── BERTScore (RoBERTa-large)
      ↓
6. Compute LLM Quality Metrics
   ├── Relevancy
   ├── Completeness
   └── Coherence
      ↓
7. Merge & Save Results
```

---

## Example Results

```
Query: "What caused customer frustration?"

Retrieval Metrics:
  - Recall: 0.75
  - MRR: 0.50
  - Hit@K: 1
  - Mean Max Cosine Sim: 0.82

NLP Metrics:
  - BLEU: 0.34
  - ROUGE-L: 0.58
  - BERTScore F1: 0.72

LLM Metrics:
  - Relevancy: 0.85
  - Completeness: 0.78
  - Coherence: 0.92
```

---

## Performance

| Metric | Value |
|--------|-------|
| Embedding Speed | ~1000 texts/minute (OpenAI) |
| BERTScore | ~10 samples/second (GPU) |
| LLM Scoring | ~2-3 seconds/sample |

---

## License

This project is provided as-is for educational and research purposes.
