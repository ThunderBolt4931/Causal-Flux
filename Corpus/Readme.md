# Corpus Generation Module

![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-blue?logo=google)
![Async](https://img.shields.io/badge/Processing-Async-green)

## Overview

This module processes raw call transcripts to extract structured metadata using **Google Gemini AI**. It transforms unstructured conversations into richly annotated data suitable for downstream RAG and causal analysis.

### Why This Matters

Direct statistical or embedding-based analysis of raw transcripts is insufficient for causal reasoning. This system uses a **hybrid approach**:
- **LLM-based structural extraction** for event, driver, and summary metadata
- **Turn-level sentiment scoring** for emotional trajectory analysis

---

## Files

| File | Description |
|------|-------------|
| `Corpus_Generation.py` | Main async processing pipeline using Gemini API |
| `system_prompt.txt` | LLM system prompt defining extraction schema |
| `Default_corpus.json` | Raw input transcripts (source data) |
| `corpus.json` | **Output:** Processed corpus with metadata |

---

## Extracted Metadata

For each transcript, the system extracts:

### 1. Completed Fields
```json
{
  "domain": "Telecom",
  "intent": "Billing Inquiry",
  "reason_for_the_call": "Customer disputing monthly charges"
}
```

### 2. Call Summary
Short narrative capturing major events in the conversation.

### 3. Outcome
Resolution or terminal state of the call (e.g., "Resolved", "Escalated").

### 4. Predefined Interaction Drivers
Mapped from a fixed taxonomy of call-center behaviors:
```json
{
  "driver": "billing_dispute",
  "dialogue_pair_index": 3
}
```

### 5. Identified Interaction Drivers
Newly discovered drivers not in the predefined taxonomy:
```json
{
  "keyword": "service_comparison",
  "definition": "Customer comparing service with competitors",
  "dialogue_pair_index": 5
}
```

---

## Configuration

Edit the following in `Corpus_Generation.py`:

```python
# API Configuration
API_KEY = os.environ.get('GEMINI_API_KEY', 'your-api-key')

# Input/Output Files
TRANSCRIPT_FILE = 'Default_corpus.json'
SYSPROMPT_FILE = 'system_prompt.txt'
TEMP_OUTPUT_FILE = 'processed_transcripts.jsonl'

# Processing Settings
NUM_WORKERS = 5          # Concurrent API calls
LIMIT = 1000             # Max transcripts (None for all)
```

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Set API Key
```bash
# Windows
set GEMINI_API_KEY=your-api-key

# Linux/Mac
export GEMINI_API_KEY=your-api-key
```

### Run Processing
```bash
python Corpus_Generation.py
```

### Output
The script generates:
1. `processed_transcripts.jsonl` - Intermediate JSONL output
2. `corpus.json` - Final JSON array (after `rephraser()` function)

---

## Output Schema

```json
{
  "transcript_id": "T001",
  "domain": "Telecom",
  "intent": "Billing Inquiry",
  "reason_for_call": "Dispute over charges",
  "turns": [...],
  "metadata": {
    "completed_fields": {
      "domain": "Telecom",
      "intent": "Billing Inquiry",
      "reason_for_the_call": "..."
    },
    "call_summary": "Customer called about...",
    "outcome": "Resolved with credit",
    "predefined_interaction_drivers": [...],
    "identified_interaction_drivers": [...]
  }
}
```

---

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~5-10 transcripts/second (5 workers) |
| API Model | `gemini-2.0-flash-exp` |
| Temperature | 0.1 (deterministic output) |

---

## License

This project is provided as-is for educational and research purposes.
