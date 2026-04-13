import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
import math
import textwrap

# --- 1. MOCK DATA & RETRIEVAL LOGIC ---

def get_mock_data():
    """
    Generates mock data matching the user's structure.
    """
    return [
        # Group 1: Biology
        {
            "type": "L2Cluster",
            "field": "biology",
            "id": "bio_L2_0",
            "name": "Genomics and Computational Biology", 
            "description": "Study of genomes.",
            "parent_id": "summary_L1_8",
            "member_ids": ["id"] * 150
        },
        {
            "type": "L2Cluster",
            "field": "biology",
            "id": "bio_L2_1",
            "name": "Proteomics",
            "description": "Study of proteins.",
            "parent_id": "summary_L1_8",
            "member_ids": ["id"] * 80
        },
        {
            "type": "L2Cluster",
            "field": "biology",
            "id": "bio_L2_2",
            "name": "Bioinformatics",
            "description": "Comp bio.",
            "parent_id": "summary_L1_8",
            "member_ids": ["id"] * 60
        },
        # Group 2: Physics
        {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_0",
            "name": "Quantum Mechanics and Computing", 
            "description": "Physics at atomic scale.",
            "parent_id": "summary_L1_12",
            "member_ids": ["id"] * 200
        },
        {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_1",
            "name": "Relativity",
            "description": "Spacetime physics.",
            "parent_id": "summary_L1_12",
            "member_ids": ["id"] * 100
        },
         {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_2",
            "name": "Thermodynamics",
            "description": "Heat and energy.",
            "parent_id": "summary_L1_12",
            "member_ids": ["id"] * 50
        },
        # Group 3: History (Smaller group)
        {
            "type": "L2Cluster",
            "field": "history",
            "id": "hist_L2_0",
            "name": "Ancient Rome",
            "description": "Roman Empire history.",
            "parent_id": "summary_L1_5",
            "member_ids": ["id"] * 40
        },
        {
            "type": "L2Cluster",
            "field": "history",
            "id": "hist_L2_1",
            "name": "Greek Mythology",
            "description": "Myths and legends.",
            "parent_id": "summary_L1_5",
            "member_ids": ["id"] * 30
        }
    ]

def retrieve(query, all_data):
    """
    Filters L2 clusters related to the query.
    """
    query = query.lower()
    results = []
    for item in all_data:
        if item.get("type") != "L2Cluster":
            continue
            
        text_content = (
            item.get("name", "") + 
            item.get("description", "") + 
            item.get("parent_id", "") +
            item.get("field", "")
        ).lower()
        
        if query in text_content:
            results.append(item)
    return results

# --- 2. BUBBLE PACKING LOGIC ---

def pack_circles(radii_with_info, padding=0):
    """
    Simple greedy circle packing using spiral search.
    """
    sorted_circles = sorted(radii_with_info, key=lambda x: x['r'], reverse=True)
    packed = []

    for circle in sorted_circles:
        r = circle['r']
        
        if not packed:
            packed.append({**circle, 'x': 0, 'y': 0})
            continue

        placed = False
        angle_step = 0.1 
        radial_step = r * 0.1 
        current_angle = 0
        current_dist = r + padding 
        
        max_dist = sum(c['r'] for c in sorted_circles) * 2 
        
        while not placed and current_dist < max_dist:
            x = current_dist * math.cos(current_angle)
            y = current_dist * math.sin(current_angle)
            
            collision = False
            for p in packed:
                dist_sq = (x - p['x'])**2 + (y - p['y'])**2
                min_dist = (r + p['r'] + padding) 
                
                if dist_sq < (min_dist * min_dist) - 0.001: 
                    collision = True
                    break
            
            if not collision:
                packed.append({**circle, 'x': x, 'y': y})
                placed = True
            
            current_angle += angle_step
            current_dist += (radial_step * angle_step / (2 * math.pi))
            
    return packed

def process_bubbles(clusters):
    """
    Hierarchical packing.
    """
    grouped = {}
    for c in clusters:
        pid = c['parent_id']
        if pid not in grouped:
            grouped[pid] = []
        
        # L2 Radius proportional to sqrt(area/members)
        radius = math.sqrt(len(c['member_ids']))
        grouped[pid].append({'data': c, 'r': radius})

    # Pack L2s inside L1s
    l1_bubbles = []
    
    for pid, children in grouped.items():
        packed_children = pack_circles(children, padding=1.5)
        
        max_extent = 0
        for child in packed_children:
            dist = math.sqrt(child['x']**2 + child['y']**2)
            extent = dist + child['r']
            if extent > max_extent:
                max_extent = extent
        
        l1_radius = max_extent * 1.1 
        
        l1_bubbles.append({
            'id': pid,
            'r': l1_radius,
            'children': packed_children,
            'num_members': sum(len(c['data']['member_ids']) for c in children)
        })

    # Pack L1s into Global Frame
    final_layout = pack_circles(l1_bubbles, padding=10.0)
    
    for l1 in final_layout:
        cx, cy = l1['x'], l1['y']
        for child in l1['children']:
            child['abs_x'] = cx + child['x']
            child['abs_y'] = cy + child['y']
            
    return final_layout

# --- 3. PLOTTING FUNCTION (UPDATED PALETTE) ---

def plot_bubbles(l1_layout):
    """
    Draws the disjoint bubble packing with 'Amethyst & Rose' palette.
    """
    if not l1_layout:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # --- PALETTE DEFINITION ---
    # Darkest shade for Text & Outlines
    MIDNIGHT_VIOLET = '#2A0A3B'
    
    # Gradient Cycle for Clusters (Purple -> Pink)
    GRADIENT_PALETTE = [
        '#651FFF', # Royal Purple
        '#AA00FF', # Electric Orchid
        '#D500F9', # Berry Pink
        '#FF4081'  # Neon Rose
    ]
    
    # Find plot bounds
    all_x = []
    all_y = []
    for l1 in l1_layout:
        all_x.append(l1['x'] + l1['r'])
        all_x.append(l1['x'] - l1['r'])
        all_y.append(l1['y'] + l1['r'])
        all_y.append(l1['y'] - l1['r'])
        
    margin = 15
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw
    for i, l1 in enumerate(l1_layout):
        # Cycle through the gradient
        color_idx = i % len(GRADIENT_PALETTE)
        parent_base_hex = GRADIENT_PALETTE[color_idx]
        
        # Create Lighter Tint for L1 Parent Bubble (10% Opacity)
        rgb = mcolors.to_rgb(parent_base_hex)
        l1_fill_color = (*rgb, 0.1) 
        
        # 1. Draw L1 Parent Bubble
        l1_circle = patches.Circle(
            (l1['x'], l1['y']), 
            l1['r'], 
            linewidth=2, 
            edgecolor=parent_base_hex,  # Solid border of base color
            facecolor=l1_fill_color,    # Very transparent fill
            linestyle='--'
        )
        ax.add_patch(l1_circle)
        
        # L1 Label (High Contrast Color)
        ax.text(
            l1['x'], 
            l1['y'] + l1['r'] + 2, 
            f"{l1['id']}\n({l1['num_members']})", 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontfamily='sans-serif',
            fontweight='bold', 
            color=MIDNIGHT_VIOLET # Using Midnight Violet for readability
        )

        # 2. Draw L2 Child Bubbles
        for j, l2 in enumerate(l1['children']):
            
            # Draw Circle (Use Base Color, High Opacity)
            l2_circle = patches.Circle(
                (l2['abs_x'], l2['abs_y']), 
                l2['r'], 
                linewidth=0.5, 
                edgecolor='white', 
                facecolor=parent_base_hex, 
                alpha=0.9 # Nearly solid
            )
            ax.add_patch(l2_circle)
            
            # L2 Label
            if l2['r'] > 2.5: 
                char_limit = max(8, int(l2['r'] * 2.0))
                wrapped_text = "\n".join(textwrap.wrap(l2['data']['name'], width=char_limit))
                
                txt = ax.text(
                    l2['abs_x'], 
                    l2['abs_y'], 
                    wrapped_text, 
                    ha='center', 
                    va='center', 
                    fontsize=9, 
                    fontfamily='sans-serif',
                    color='white'
                )
                # Text Outline using Midnight Violet for consistency
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground=MIDNIGHT_VIOLET)])

    plt.title("Cluster Distribution", fontsize=16, pad=20, color=MIDNIGHT_VIOLET, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    all_data = get_mock_data()
    user_query = "" 
    filtered_data = retrieve(user_query, all_data)
    
    print(f"Query returned {len(filtered_data)} clusters.")
    
    if filtered_data:
        final_layout = process_bubbles(filtered_data)
        plot_bubbles(final_layout)
    else:
        print("No data found for query.")