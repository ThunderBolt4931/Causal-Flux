# Query Generation Module

This folder contains scripts for generating causal reasoning queries from call transcript data.

## Files

| File | Description |
|------|-------------|
| `Query_Generation_Task1.py` | Generates base queries targeting primary causal relationships in call interactions |
| `Query_Generation_Task2.py` | Produces follow-up queries for multi-turn analysis, building on the base queries |

---

## Pipelines Used in Query Answering

Queries are answered using the two best-performing pipelines:

1. **Cluster Masked Interaction Driven Graph Retriever** — Combines cluster-level filtering with interaction-driver-based graph traversal for high-precision retrieval.
2. **Interaction Driven Graph Retriever** — Uses interaction drivers as seed nodes for Personalized PageRank scoring without cluster masking.

Other pipelines were excluded due to subpar output relevance.

---

## Usage

Navigate to the `Queries` folder and run:

```bash
python Query_Generation_Task1.py
python Query_Generation_Task2.py
```

---

## License

This project is provided as-is for educational and research purposes.