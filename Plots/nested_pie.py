import matplotlib.pyplot as plt
import numpy as np

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
            "name": "Genomics",
            "description": "Study of genomes.",
            "embedding": [],
            "parent_id": "Biology (L1)",
            "member_ids": ["id"] * 150
        },
        {
            "type": "L2Cluster",
            "field": "biology",
            "id": "bio_L2_1",
            "name": "Proteomics",
            "description": "Study of proteins.",
            "embedding": [],
            "parent_id": "Biology (L1)",
            "member_ids": ["id"] * 80 
        },
        # Group 2: Physics
        {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_0",
            "name": "Quantum",
            "description": "Physics at atomic scale.",
            "embedding": [],
            "parent_id": "Physics (L1)",
            "member_ids": ["id"] * 200 
        },
        {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_1",
            "name": "Relativity",
            "description": "Spacetime physics.",
            "embedding": [],
            "parent_id": "Physics (L1)",
            "member_ids": ["id"] * 100 
        },
         {
            "type": "L2Cluster",
            "field": "physics",
            "id": "phys_L2_2",
            "name": "Thermodynamics",
            "description": "Heat and energy.",
            "embedding": [],
            "parent_id": "Physics (L1)",
            "member_ids": ["id"] * 50 
        },
        # Group 3: History (Unrelated)
        {
            "type": "L2Cluster",
            "field": "history",
            "id": "hist_L2_0",
            "name": "Ancient Rome",
            "description": "Roman Empire history.",
            "embedding": [],
            "parent_id": "History (L1)",
            "member_ids": ["id"] * 20
        }
    ]

def retrieve(query, all_data):
    query = query.lower()
    results = []
    for item in all_data:
        if item.get("type") != "L2Cluster":
            continue
        text_content = (
            item.get("name", "") + 
            item.get("description", "") + 
            item.get("parent_id", "")
        ).lower()
        
        if query in text_content:
            results.append(item)
    return results

# --- 2. DATA PROCESSING ---

def process_data_for_plotting(clusters):
    # 1. Group by Parent ID
    grouped = {}
    for c in clusters:
        pid = c['parent_id']
        if pid not in grouped:
            grouped[pid] = {'children': [], 'total_size': 0}
        
        size = len(c['member_ids'])
        grouped[pid]['children'].append({
            'data': c,
            'size': size
        })
        grouped[pid]['total_size'] += size

    sorted_parent_ids = sorted(grouped.keys())

    # 3. Flatten into ordered lists for plotting
    l2_slices = [] # Inner pie
    l1_slices = [] # Outer ring
    
    total_population = sum(g['total_size'] for g in grouped.values())
    if total_population == 0:
        return [], []

    current_angle = 0 # Starting angle in degrees
    
    # --- CUSTOM PALETTE: Amethyst & Rose ---
    # We remove the very darkest color (#2A0A3B) from the slices 
    # to ensure the chart remains bright and readable, 
    # reserving it for text/titles instead.
    CUSTOM_PALETTE = [
        '#651FFF', # Royal Purple
        '#AA00FF', # Electric Orchid
        '#D500F9', # Berry Pink
        '#FF4081'  # Neon Rose
    ]
    
    for p_idx, pid in enumerate(sorted_parent_ids):
        group = grouped[pid]
        
        # Pick color looping through the palette
        base_color = CUSTOM_PALETTE[p_idx % len(CUSTOM_PALETTE)]
        
        # --- Outer Slice (L1 Parent) ---
        parent_ratio = group['total_size'] / total_population
        parent_angle_width = parent_ratio * 360
        
        l1_slices.append({
            'name': pid,
            'size': group['total_size'],
            'start_angle': current_angle,
            'end_angle': current_angle + parent_angle_width,
            'color': base_color 
        })
        
        # --- Inner Slices (L2 Children) ---
        child_start_angle = current_angle
        num_children = len(group['children'])
        
        for i, child in enumerate(group['children']):
            child_ratio = child['size'] / group['total_size']
            child_angle_width = child_ratio * parent_angle_width
            
            # Create semi-transparent variation for children
            # Range alpha from 0.4 (lighter) to 0.7 (darker)
            if num_children > 1:
                alpha_val = 0.4 + (0.3 * (i / (num_children - 1)))
            else:
                alpha_val = 0.5

            l2_slices.append({
                'name': child['data']['name'],
                'size': child['size'],
                'start_angle': child_start_angle,
                'end_angle': child_start_angle + child_angle_width,
                'color': base_color,  # Inherit parent color
                'alpha': alpha_val    # Apply transparency
            })
            
            child_start_angle += child_angle_width
            
        current_angle += parent_angle_width
        
    return l1_slices, l2_slices

# --- 3. PLOTTING FUNCTION ---

def plot_nested_pie(l1_slices, l2_slices):
    """
    Draws the nested pie chart using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_axis_off()
    
    # --- PALETTE CONSTANT ---
    MIDNIGHT_VIOLET = '#2A0A3B' # Dark color for Text/Boxes
    
    # --- CONFIGURATION ---
    split_radius = 0.7  # Inner Radius
    
    # --- Plot Inner Pie (L2 Clusters) ---
    for item in l2_slices:
        theta = np.radians([item['start_angle'], item['end_angle']])
        width = np.radians(item['end_angle'] - item['start_angle'])
        center = np.radians(item['start_angle']) + width/2
        
        ax.bar(
            x=center, 
            height=split_radius,   # 0 to 0.7
            width=width, 
            bottom=0.0, 
            color=item['color'], 
            alpha=item['alpha'],   # Semi-transparent
            edgecolor='white',
            linewidth=1
        )
        
        # Label Inner
        if width > np.radians(5): 
            # Position label at 60% of the inner radius
            label_pos = split_radius * 0.6 
            
            ax.text(center, label_pos, item['name'], 
                    ha='center', va='center', fontsize=9, 
                    rotation=0, # Horizontal
                    color='white', fontweight='bold',
                    # Box uses Midnight Violet for theme consistency
                    bbox=dict(facecolor=MIDNIGHT_VIOLET, alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'))

    # --- Plot Outer Ring (L1 Clusters) ---
    for item in l1_slices:
        width = np.radians(item['end_angle'] - item['start_angle'])
        center = np.radians(item['start_angle']) + width/2
        
        ax.bar(
            x=center, 
            height=(1.0 - split_radius), # 0.3 thickness (0.7 to 1.0)
            width=width, 
            bottom=split_radius, 
            color=item['color'], 
            alpha=1.0, # Solid color for the parent ring
            edgecolor='white',
            linewidth=2
        )
        
        # Label Outer
        # Position label in the middle of the outer ring
        label_pos = split_radius + (1.0 - split_radius) / 2
        
        ax.text(center, label_pos, item['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold', 
                rotation=0, # Horizontal
                color='white',
                # Box uses Midnight Violet with slightly higher opacity
                bbox=dict(facecolor=MIDNIGHT_VIOLET, alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title("Cluster Breakdown", y=1.05, fontsize=16, color=MIDNIGHT_VIOLET, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    all_data = get_mock_data()
    filtered_data = retrieve("", all_data) 
    
    if filtered_data:
        l1_pie, l2_pie = process_data_for_plotting(filtered_data)
        plot_nested_pie(l1_pie, l2_pie)
    else:
        print("No data found.")