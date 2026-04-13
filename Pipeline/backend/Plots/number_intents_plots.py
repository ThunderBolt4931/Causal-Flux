import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def count_driver_occurrences(file_path, target_drivers, target_transcript_ids):
    """
    Counts occurrences of specific interaction drivers within specific transcripts.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return {}
    
    allowed_drivers_set = set(target_drivers)
    allowed_ids_set = set(target_transcript_ids)
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
                
    return result_map

def plot_driver_occurrences(file_path, target_drivers, target_ids):
    """
    Generates a horizontal bar chart showing the total frequency of interaction drivers
    using the Amethyst & Rose palette.
    """

    counts_data = count_driver_occurrences(file_path, target_drivers, target_ids)

    if not counts_data:
        print("No data found or file error.")
        return

    drivers_list = []
    total_counts = []

    for driver in target_drivers: 
        transcript_counts = counts_data.get(driver, {})
        total = sum(transcript_counts.values())
        
        drivers_list.append(driver)
        total_counts.append(total)

    MIDNIGHT_VIOLET = '#2A0A3B'
    
    hex_palette = [
        '#651FFF', # Royal Purple
        '#AA00FF', # Electric Orchid
        '#D500F9', # Berry Pink
        '#FF4081'  # Neon Rose
    ]
    bar_fills = []
    bar_edges = []
    
    for i in range(len(drivers_list)):
        color_hex = hex_palette[i % len(hex_palette)]
        rgba = mcolors.to_rgba(color_hex, alpha=0.4)
        
        bar_fills.append(rgba)
        bar_edges.append(color_hex)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        drivers_list, 
        total_counts, 
        color=bar_fills, 
        edgecolor=bar_edges,
        linewidth=2 
    )
    plt.xlabel('Total Frequency', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    plt.ylabel('Interaction Drivers', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    plt.title('Total Frequency of Interaction Drivers', fontsize=14, fontweight='bold', color=MIDNIGHT_VIOLET)
    plt.grid(axis='x', linestyle=':', alpha=0.6, color='#651FFF')
    ax = plt.gca()
    ax.tick_params(axis='x', colors=MIDNIGHT_VIOLET)
    ax.tick_params(axis='y', colors=MIDNIGHT_VIOLET)
    if total_counts:
        max_val = max(total_counts)
        plt.xticks(range(int(max_val) + 2))
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_color = bar_edges[i]
        
        plt.text(
            width + 0.1,    
            bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', 
            va='center', 
            fontweight='bold',
            color=label_color
        )

    plt.tight_layout()
    plt.show()