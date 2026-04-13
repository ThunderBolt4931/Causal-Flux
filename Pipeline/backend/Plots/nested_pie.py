import matplotlib.pyplot as plt
import numpy as np


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

def process_data_for_plotting(clusters):
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

    l2_slices = [] 
    l1_slices = [] 
    
    total_population = sum(g['total_size'] for g in grouped.values())
    if total_population == 0:
        return [], []

    current_angle = 0
    CUSTOM_PALETTE = [
        '#651FFF', 
        '#AA00FF', 
        '#D500F9', 
        '#FF4081'  
    ]
    
    for p_idx, pid in enumerate(sorted_parent_ids):
        group = grouped[pid]
        base_color = CUSTOM_PALETTE[p_idx % len(CUSTOM_PALETTE)]
        parent_ratio = group['total_size'] / total_population
        parent_angle_width = parent_ratio * 360
        
        l1_slices.append({
            'name': pid,
            'size': group['total_size'],
            'start_angle': current_angle,
            'end_angle': current_angle + parent_angle_width,
            'color': base_color 
        })
        child_start_angle = current_angle
        num_children = len(group['children'])
        
        for i, child in enumerate(group['children']):
            child_ratio = child['size'] / group['total_size']
            child_angle_width = child_ratio * parent_angle_width
            if num_children > 1:
                alpha_val = 0.4 + (0.3 * (i / (num_children - 1)))
            else:
                alpha_val = 0.5

            l2_slices.append({
                'name': child['data']['name'],
                'size': child['size'],
                'start_angle': child_start_angle,
                'end_angle': child_start_angle + child_angle_width,
                'color': base_color,  
                'alpha': alpha_val    
            })
            
            child_start_angle += child_angle_width
            
        current_angle += parent_angle_width
        
    return l1_slices, l2_slices

def plot_nested_pie(l1_slices, l2_slices):
    """
    Draws the nested pie chart using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_axis_off()
    MIDNIGHT_VIOLET = '#2A0A3B' 

    split_radius = 0.7  

    for item in l2_slices:
        theta = np.radians([item['start_angle'], item['end_angle']])
        width = np.radians(item['end_angle'] - item['start_angle'])
        center = np.radians(item['start_angle']) + width/2
        
        ax.bar(
            x=center, 
            height=split_radius,   
            width=width, 
            bottom=0.0, 
            color=item['color'], 
            alpha=item['alpha'],  
            edgecolor='white',
            linewidth=1
        )

        if width > np.radians(5): 
            label_pos = split_radius * 0.6 
            
            ax.text(center, label_pos, item['name'], 
                    ha='center', va='center', fontsize=9, 
                    rotation=0,
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=MIDNIGHT_VIOLET, alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'))
    for item in l1_slices:
        width = np.radians(item['end_angle'] - item['start_angle'])
        center = np.radians(item['start_angle']) + width/2
        
        ax.bar(
            x=center, 
            height=(1.0 - split_radius), 
            width=width, 
            bottom=split_radius, 
            color=item['color'], 
            alpha=1.0, 
            edgecolor='white',
            linewidth=2
        )
        
        label_pos = split_radius + (1.0 - split_radius) / 2
        
        ax.text(center, label_pos, item['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold', 
                rotation=0, 
                color='white',
                bbox=dict(facecolor=MIDNIGHT_VIOLET, alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title("Cluster Breakdown", y=1.05, fontsize=16, color=MIDNIGHT_VIOLET, fontweight='bold')
    plt.tight_layout()
    plt.show()
