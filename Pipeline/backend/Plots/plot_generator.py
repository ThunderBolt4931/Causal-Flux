"""
Plot Generator Utility
Generates matplotlib plots as base64-encoded PNG images for API responses.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import math
import textwrap
import io
import base64
import json
import collections


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_intents_bar_chart(file_path: str, target_drivers: list, target_transcript_ids: list) -> str:
    """
    Generate vertical bar chart showing number of calls per interaction driver.
    Returns base64-encoded PNG image.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return ""
    
    allowed_drivers_set = set(target_drivers)
    allowed_ids_set = set(target_transcript_ids)
    
    driver_map = collections.defaultdict(list)
    
    for call in data:
        transcript_id = call.get('transcript_id')
        
        if transcript_id not in allowed_ids_set:
            continue
            
        metadata = call.get('metadata', {})
        predefined_drivers = metadata.get('predefined_interaction_drivers', [])
        
        found_drivers_in_call = set()
        for driver_info in predefined_drivers:
            driver_name = driver_info.get('driver')
            
            if driver_name and driver_name in allowed_drivers_set:
                found_drivers_in_call.add(driver_name)
        
        for driver_name in found_drivers_in_call:
            driver_map[driver_name].append(transcript_id)
    
    if not driver_map:
        return ""
    
    drivers = list(driver_map.keys())
    call_counts = [len(ids) for ids in driver_map.values()]
    hex_palette = [
        '#2A0A3B', '#651FFF', '#AA00FF', '#D500F9', '#FF4081'
    ]
    
    bar_fills = []
    bar_edges = []
    
    for i in range(len(drivers)):
        color_hex = hex_palette[i % len(hex_palette)]
        rgba = mcolors.to_rgba(color_hex, alpha=0.4)
        bar_fills.append(rgba)
        bar_edges.append(color_hex)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(
        drivers, 
        call_counts, 
        color=bar_fills, 
        edgecolor=bar_edges, 
        linewidth=2
    )
    
    text_color = '#2A0A3B'
    
    ax.set_xlabel('Interaction Drivers (Intents)', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Number of Calls', fontsize=12, fontweight='bold', color=text_color)
    ax.set_title('Number of Calls per Interaction Driver', fontsize=14, fontweight='bold', color=text_color)
    
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle=':', alpha=0.6, color='#651FFF')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_color = bar_edges[i]
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom',
                 fontweight='bold',
                 color=label_color)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_frequency_chart(file_path: str, target_drivers: list, target_ids: list) -> str:
    """
    Generate horizontal bar chart showing total frequency of interaction drivers.
    Returns base64-encoded PNG image.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return ""
    
    allowed_drivers_set = set(target_drivers)
    allowed_ids_set = set(target_ids)
    
    result_map = {driver: {} for driver in target_drivers}
    
    for call in data:
        transcript_id = call.get('transcript_id')
        
        if transcript_id not in allowed_ids_set:
            continue
            
        metadata = call.get('metadata', {})
        predefined_drivers = metadata.get('predefined_interaction_drivers', [])
        
        for driver_entry in predefined_drivers:
            driver_name = driver_entry.get('driver')
            
            if driver_name in allowed_drivers_set:
                if transcript_id not in result_map[driver_name]:
                    result_map[driver_name][transcript_id] = 0
                
                result_map[driver_name][transcript_id] += 1
    
    if not result_map:
        return ""

    drivers_list = []
    total_counts = []
    
    for driver in target_drivers:
        transcript_counts = result_map.get(driver, {})
        total = sum(transcript_counts.values())
        
        drivers_list.append(driver)
        total_counts.append(total)
    MIDNIGHT_VIOLET = '#2A0A3B'
    
    hex_palette = [
        '#651FFF', '#AA00FF', '#D500F9', '#FF4081'
    ]
    
    bar_fills = []
    bar_edges = []
    
    for i in range(len(drivers_list)):
        color_hex = hex_palette[i % len(hex_palette)]
        rgba = mcolors.to_rgba(color_hex, alpha=0.4)
        
        bar_fills.append(rgba)
        bar_edges.append(color_hex)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(
        drivers_list, 
        total_counts, 
        color=bar_fills, 
        edgecolor=bar_edges,
        linewidth=2
    )
    
    ax.set_xlabel('Total Frequency', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    ax.set_ylabel('Interaction Drivers', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    ax.set_title('Total Frequency of Interaction Drivers', fontsize=14, fontweight='bold', color=MIDNIGHT_VIOLET)
    
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='#651FFF')
    
    ax.tick_params(axis='x', colors=MIDNIGHT_VIOLET)
    ax.tick_params(axis='y', colors=MIDNIGHT_VIOLET)
    
    if total_counts:
        max_val = max(total_counts)
        ax.set_xticks(range(int(max_val) + 2))
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_color = bar_edges[i]
        
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            va='center',
            fontweight='bold',
            color=label_color
        )
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cluster_pie(cluster_file: str, retrieved_transcript_ids: list) -> str:
    """
    Generate nested pie chart showing L1/L2 cluster breakdown.
    Returns base64-encoded PNG image.
    """
    try:
        with open(cluster_file, 'r', encoding='utf-8') as f:
            clusterings = json.load(f)
    except FileNotFoundError:
        return ""
    
    l2_clusters = []
    for c in clusterings:
        if c.get('type') == 'L2Cluster':
            member_ids = set(c.get('member_ids', []))
            if member_ids.intersection(set(retrieved_transcript_ids)):
                l2_clusters.append(c)
    
    if not l2_clusters:
        return ""
    
    grouped = {}
    for c in l2_clusters:
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
        return ""
    
    current_angle = 0
    
    CUSTOM_PALETTE = [
        '#651FFF', '#AA00FF', '#D500F9', '#FF4081'
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
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_axis_off()
    
    MIDNIGHT_VIOLET = '#2A0A3B'
    split_radius = 0.7
    
    # Plot inner pie (L2)
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
    
    # Plot outer ring (L1)
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
    return _fig_to_base64(fig)


def generate_bubble_chart(cluster_file: str, retrieved_transcript_ids: list) -> str:
    """
    Generate bubble packing visualization showing cluster distribution.
    Returns base64-encoded PNG image.
    """
    try:
        with open(cluster_file, 'r', encoding='utf-8') as f:
            clusterings = json.load(f)
    except FileNotFoundError:
        return ""
    
    # Filter L2 clusters
    l2_clusters = []
    for c in clusterings:
        if c.get('type') == 'L2Cluster':
            member_ids = set(c.get('member_ids', []))
            if member_ids.intersection(set(retrieved_transcript_ids)):
                l2_clusters.append(c)
    
    if not l2_clusters:
        return ""
    
    # Bubble packing logic
    def pack_circles(radii_with_info, padding=0):
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
    
    # Process bubbles
    grouped = {}
    for c in l2_clusters:
        pid = c['parent_id']
        if pid not in grouped:
            grouped[pid] = []
        
        radius = math.sqrt(len(c['member_ids']))
        grouped[pid].append({'data': c, 'r': radius})
    
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
    
    final_layout = pack_circles(l1_bubbles, padding=10.0)
    
    for l1 in final_layout:
        cx, cy = l1['x'], l1['y']
        for child in l1['children']:
            child['abs_x'] = cx + child['x']
            child['abs_y'] = cy + child['y']
    if not final_layout:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    MIDNIGHT_VIOLET = '#2A0A3B'
    
    GRADIENT_PALETTE = [
        '#651FFF', '#AA00FF', '#D500F9', '#FF4081'
    ]
    
    all_x = []
    all_y = []
    for l1 in final_layout:
        all_x.append(l1['x'] + l1['r'])
        all_x.append(l1['x'] - l1['r'])
        all_y.append(l1['y'] + l1['r'])
        all_y.append(l1['y'] - l1['r'])
        
    margin = 15
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    for i, l1 in enumerate(final_layout):
        color_idx = i % len(GRADIENT_PALETTE)
        parent_base_hex = GRADIENT_PALETTE[color_idx]
        
        rgb = mcolors.to_rgb(parent_base_hex)
        l1_fill_color = (*rgb, 0.1)
        
        l1_circle = patches.Circle(
            (l1['x'], l1['y']),
            l1['r'],
            linewidth=2,
            edgecolor=parent_base_hex,
            facecolor=l1_fill_color,
            linestyle='--'
        )
        ax.add_patch(l1_circle)
        
        ax.text(
            l1['x'],
            l1['y'] + l1['r'] + 2,
            f"{l1['id']}\n({l1['num_members']})",
            ha='center',
            va='bottom',
            fontsize=11,
            fontfamily='sans-serif',
            fontweight='bold',
            color=MIDNIGHT_VIOLET
        )
        
        for j, l2 in enumerate(l1['children']):
            l2_circle = patches.Circle(
                (l2['abs_x'], l2['abs_y']),
                l2['r'],
                linewidth=0.5,
                edgecolor='white',
                facecolor=parent_base_hex,
                alpha=0.9
            )
            ax.add_patch(l2_circle)
            
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
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground=MIDNIGHT_VIOLET)])
    
    plt.title("Cluster Distribution", fontsize=16, pad=20, color=MIDNIGHT_VIOLET, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)
