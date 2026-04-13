# Visualization Module

![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-green)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

## Overview

This module generates **visual analytics** for RAG query results, providing insights into:

- **Sentiment trajectories** across conversations
- **Intent/driver distributions** in retrieved transcripts
- **Cluster distributions** for query results
- **Interactive bubble charts** for cluster visualization

---

## Files

| File | Description |
|------|-------------|
| `final_plots.py` | Sentiment curve visualizations (single + aggregate) |
| `bubbles.py` | Hierarchical bubble packing charts |
| `intents_plots.py` | Intent/driver bar charts |
| `nested_pie.py` | Nested pie chart for cluster breakdown |
| `number_intents_plots.py` | Driver frequency histograms |

---

## Visualization Types

### 1. Sentiment Progression
**File:** `final_plots.py`

Shows how sentiment evolves throughout conversations:
- Individual transcript curves (top 10)
- Aggregate trend with positive/negative fill

```python
from final_plots import plot_sentiment_curves, plot_average_sentiment_trend

plot_sentiment_curves(results, id_to_transcript)
plot_average_sentiment_trend(results, id_to_transcript)
```

### 2. Hierarchical Bubble Chart
**File:** `bubbles.py`

Displays cluster hierarchy as packed circles:
- L1 clusters as outer bubbles
- L2 clusters nested inside
- Size proportional to member count

```python
from bubbles import retrieve, process_bubbles, plot_bubbles

filtered = retrieve("billing", all_data)
layout = process_bubbles(filtered)
plot_bubbles(layout)
```

### 3. Intent Distribution
**File:** `intents_plots.py`

Bar charts showing driver/intent frequency:
- Counts per driver across retrieved transcripts
- Filtered by query-identified intents

### 4. Cluster Breakdown
**File:** `nested_pie.py`

Nested pie showing:
- Outer ring: L1 cluster distribution
- Inner ring: L2 cluster breakdown

### 5. Driver Frequency
**File:** `number_intents_plots.py`

Histogram of how many drivers appear per transcript.

---

## Color Palette

The visualizations use the **Amethyst & Rose** palette:

| Color | Hex | Usage |
|-------|-----|-------|
| Midnight Violet | `#2A0A3B` | Text, outlines |
| Royal Purple | `#651FFF` | Primary |
| Electric Orchid | `#AA00FF` | Secondary |
| Berry Pink | `#D500F9` | Tertiary |
| Neon Rose | `#FF4081` | Accent |

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Sentiment Curves
```python
import json
from final_plots import plot_sentiment_curves

# Load data
with open("corpus.json") as f:
    data = json.load(f)
id_to_transcript = {t["transcript_id"]: t for t in data}

# Mock results (from RAG query)
results = [
    {"rank": 1, "transcript_id": "T001"},
    {"rank": 2, "transcript_id": "T002"},
    # ...
]

plot_sentiment_curves(results, id_to_transcript)
```

### Bubble Charts
```python
from bubbles import get_mock_data, retrieve, process_bubbles, plot_bubbles

# Load cluster data
all_data = get_mock_data()  # or load from clustered_transcripts.json

# Filter by query
filtered = retrieve("billing", all_data)

# Generate layout
layout = process_bubbles(filtered)

# Render
plot_bubbles(layout)
```

---

## Output Examples

### Sentiment Progression
```
      Sentiment
    1.0 │    ╭──╮
    0.5 │   ╱    ╲
    0.0 │──╯      ╲────
   -0.5 │          ╲   ╱
   -1.0 │           ╰─╯
        └─────────────────
          Turn 1  5  10  15
```

### Bubble Hierarchy
```
    ┌───────────────────────────────┐
    │  L1: Billing Issues (150)     │
    │  ┌─────────┐  ┌─────────┐     │
    │  │ Payment │  │ Refund  │     │
    │  │ (80)    │  │ (70)    │     │
    │  └─────────┘  └─────────┘     │
    └───────────────────────────────┘
```

---

## Integration with Backend

The `Pipeline/backend/Plots/plot_generator.py` uses these modules to generate Base64-encoded images for the frontend:

```python
def generate_cluster_pie(clusterings_file, transcript_ids):
    # Generate plot
    fig = create_nested_pie(...)
    
    # Encode as Base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >=3.7.0 | Core plotting |
| seaborn | >=0.12.0 | Statistical viz |
| numpy | >=1.24.0 | Numerical ops |
| scipy | >=1.10.0 | Interpolation |

---

## License

This project is provided as-is for educational and research purposes.
