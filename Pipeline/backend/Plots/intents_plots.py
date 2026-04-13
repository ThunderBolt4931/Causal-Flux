import json
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def filter_transcripts_by_driver(file_path, target_drivers, target_transcript_ids):
    """
    Retrieves transcripts filtered by specific interaction drivers and transcript IDs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}
    
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
            
    return dict(driver_map)

def plot_driver_counts(file_path, target_drivers, target_ids):
    driver_data = filter_transcripts_by_driver(file_path, target_drivers, target_ids)
    
    if not driver_data:
        print("No data found matching criteria.")
        return

    drivers = list(driver_data.keys())
    call_counts = [len(ids) for ids in driver_data.values()]

    hex_palette = [
        '#2A0A3B', 
        '#651FFF', 
        '#AA00FF', 
        '#D500F9', 
        '#FF4081'  
    ]

    bar_fills = []
    bar_edges = []
    
    for i in range(len(drivers)):
        color_hex = hex_palette[i % len(hex_palette)]
        rgba = mcolors.to_rgba(color_hex, alpha=0.4)
        bar_fills.append(rgba)
        bar_edges.append(color_hex)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        drivers, 
        call_counts, 
        color=bar_fills, 
        edgecolor=bar_edges, 
        linewidth=2 
    )

    text_color = '#2A0A3B'
    
    plt.xlabel('Interaction Drivers (Intents)', fontsize=12, fontweight='bold', color=text_color)
    plt.ylabel('Number of Calls', fontsize=12, fontweight='bold', color=text_color)
    plt.title('Number of Calls per Interaction Driver', fontsize=14, fontweight='bold', color=text_color)
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.6, color='#651FFF')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_color = bar_edges[i] 
        
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom',
                 fontweight='bold',
                 color=label_color)

    plt.tight_layout()
    plt.show()
