import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import math
import textwrap

def get_pipeline_data():
    """
    Represents the components found in your RAG script.
    """
    return [
        {
            "type": "Component", "group": "Configuration", "id": "cfg_01",
            "name": "Env Variables",
            "description": "Loads API Keys (OpenAI, Anthropic, Groq) and Neo4j Credentials.",
            "parent_id": "group_inputs",
            "complexity": 20
        },
        {
            "type": "Component", "group": "Configuration", "id": "cfg_02",
            "name": "JSON Dataset",
            "description": "Loads 'final_dataset.json' and maps ID_TO_TRANSCRIPT.",
            "parent_id": "group_inputs",
            "complexity": 40
        },
        {
            "type": "Component", "group": "Configuration", "id": "cfg_03",
            "name": "Arg Parser",
            "description": "Handles CLI arguments: --model, --provider, --rag type.",
            "parent_id": "group_inputs",
            "complexity": 15
        },
        {
            "type": "Component", "group": "Retrieval", "id": "rag_01",
            "name": "Neo4j Graph RAG",
            "description": "Connects to Neo4j DB. Retrieval Limit: top_k=5.",
            "parent_id": "group_retrieval",
            "complexity": 60
        },
        {
            "type": "Component", "group": "Retrieval", "id": "rag_02",
            "name": "Cosine Similarity",
            "description": "Uses embeddings matrix. Threshold: 0.425.",
            "parent_id": "group_retrieval",
            "complexity": 50
        },
        {
            "type": "Component", "group": "Retrieval", "id": "rag_03",
            "name": "BM25 Search",
            "description": "Keyword-based sparse retrieval context.",
            "parent_id": "group_retrieval",
            "complexity": 40
        },
        {
            "type": "Component", "group": "Retrieval", "id": "rag_04",
            "name": "Vanilla RAG",
            "description": "Intent classification + ChromaDB + Reranker.",
            "parent_id": "group_retrieval",
            "complexity": 55
        },

        {
            "type": "Component", "group": "Processing", "id": "proc_01",
            "name": "Query Splitter",
            "description": "Splits complex user queries into sub-queries.",
            "parent_id": "group_logic",
            "complexity": 30
        },
        {
            "type": "Component", "group": "Processing", "id": "proc_02",
            "name": "Context Router",
            "description": "Decides if retrieval is needed or if context suffices.",
            "parent_id": "group_logic",
            "complexity": 35
        },
        {
            "type": "Component", "group": "Processing", "id": "proc_03",
            "name": "LLM Inference",
            "description": "Calls run_llm with formatted context and chat history.",
            "parent_id": "group_logic",
            "complexity": 70
        },
        {
            "type": "Component", "group": "Processing", "id": "proc_04",
            "name": "CSV Output",
            "description": "Saves final answers and retrieval flags to output file.",
            "parent_id": "group_logic",
            "complexity": 25
        }
    ]


def pack_circles_spiral(circles, padding=0):
    sorted_circles = sorted(circles, key=lambda x: x['r'], reverse=True)
    packed = []
    max_dist = sum(c['r'] for c in sorted_circles) * 5 

    for circle in sorted_circles:
        r = circle['r']
        if not packed:
            packed.append({**circle, 'x': 0, 'y': 0})
            continue

        placed = False
        angle_step = 0.05 
        current_angle = 0
        current_dist = r + padding 
        
        while not placed and current_dist < max_dist:
            x = current_dist * math.cos(current_angle)
            y = current_dist * math.sin(current_angle)
            
            collision = False
            for p in packed:
                dist_sq = (x - p['x'])*2 + (y - p['y'])*2
                min_dist = (r + p['r'] + padding) 
                if dist_sq < (min_dist * min_dist) - 0.01: 
                    collision = True; break
            
            if not collision:
                packed.append({**circle, 'x': x, 'y': y})
                placed = True
            
            current_angle += angle_step
            current_dist += (r * 0.1 * angle_step / (2 * math.pi))
    return packed

def process_layout(data):
    SCALE_FACTOR = 0.8  
    
    grouped = {}
    for item in data:
        pid = item['parent_id']
        if pid not in grouped: grouped[pid] = []

        r = math.sqrt(item['complexity']) * SCALE_FACTOR
        grouped[pid].append({'data': item, 'r': r})

    l1_bubbles = []
    
    for pid, children in grouped.items():
        packed_children = pack_circles_spiral(children, padding=0.5)

        max_extent = 0
        for child in packed_children:
            dist = math.sqrt(child['x']*2 + child['y']*2)
            extent = dist + child['r']
            if extent > max_extent: max_extent = extent
        
        l1_bubbles.append({
            'id': pid,
            'name': children[0]['data']['group'], 
            'r': max_extent * 1.1,
            'children': packed_children
        })

    final_layout = pack_circles_spiral(l1_bubbles, padding=5.0)

    for l1 in final_layout:
        cx, cy = l1['x'], l1['y']
        for child in l1['children']:
            child['abs_x'] = cx + child['x']
            child['abs_y'] = cy + child['y']
            
    return final_layout


def plot_pipeline(l1_layout):
    if not l1_layout: return

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    PALETTE = ['#00B0FF', '#7C4DFF', '#00C853'] 

    all_x = []
    all_y = []
    for l1 in l1_layout:
        all_x.extend([l1['x'] + l1['r'], l1['x'] - l1['r']])
        all_y.extend([l1['y'] + l1['r'], l1['y'] - l1['r']])
        
    margin = 5

    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    chart_elements = [] 

    for i, l1 in enumerate(l1_layout):
        color_hex = PALETTE[i % len(PALETTE)]

        l1_circle = patches.Circle(
            (l1['x'], l1['y']), l1['r'], 
            linewidth=2, edgecolor=color_hex, facecolor=(0,0,0,0), linestyle='--'
        )
        ax.add_patch(l1_circle)
 
        ax.text(l1['x'], l1['y'] - l1['r'] - 2.5, l1['name'].upper(), 
                ha='center', fontsize=12, fontweight='bold', color=color_hex)

        for l2 in l1['children']:
            l2_circle = patches.Circle(
                (l2['abs_x'], l2['abs_y']), l2['r'], 
                linewidth=1, edgecolor='white', facecolor=color_hex, alpha=0.9
            )
            ax.add_patch(l2_circle)

            wrap_width = max(8, int(l2['r'] * 3.5))
            wrapped_name = "\n".join(textwrap.wrap(l2['data']['name'], width=wrap_width))
            
            font_size = max(6, l2['r'] * 0.35)

            ax.text(
                l2['abs_x'], l2['abs_y'], 
                wrapped_name, 
                ha='center', va='center', 
                fontsize=font_size, 
                fontfamily='sans-serif', 
                fontweight='bold', 
                color='white'
            )
            
            d = l2['data']
            hover_text = (
                f"COMPONENT: {d['name']}\n"
                f"ID: {d['id']}\n"
                f"----------------------\n"
                f"{d['description']}"
            )
            chart_elements.append((l2_circle, hover_text))

    annot = ax.annotate(
        "", xy=(0,0), xytext=(20,20), 
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="#333", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=10, fontfamily='monospace', color="#333"
    )
    annot.set_visible(False)
    annot.set_zorder(10)

    def update_annot(circle, text):
        annot.xy = circle.center
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            found = False
            for circle, text in reversed(chart_elements):
                cont, _ = circle.contains(event)
                if cont:
                    update_annot(circle, text)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    found = True
                    break 
            if not found and vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title("RAG Pipeline Architecture (Hover for Details)", fontsize=16, color='#333', fontweight='bold')
    plt.tight_layout()
    plt.show()
