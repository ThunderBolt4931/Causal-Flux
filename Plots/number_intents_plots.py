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
    
    # Convert lists to sets for faster lookup
    allowed_drivers_set = set(target_drivers)
    allowed_ids_set = set(target_transcript_ids)
    
    # Initialize the result dictionary with the target drivers
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
    # 1. Get the data
    counts_data = count_driver_occurrences(file_path, target_drivers, target_ids)

    if not counts_data:
        print("No data found or file error.")
        return

    # 2. Aggregate data for plotting
    drivers_list = []
    total_counts = []

    for driver in target_drivers: 
        transcript_counts = counts_data.get(driver, {})
        total = sum(transcript_counts.values())
        
        drivers_list.append(driver)
        total_counts.append(total)

    # 3. Setup Palette
    # Text Color (Darkest)
    MIDNIGHT_VIOLET = '#2A0A3B'
    
    # Gradient Palette
    hex_palette = [
        '#651FFF', # Royal Purple
        '#AA00FF', # Electric Orchid
        '#D500F9', # Berry Pink
        '#FF4081'  # Neon Rose
    ]

    # Generate lists for Fills (Semi-Transparent) and Edges (Solid)
    bar_fills = []
    bar_edges = []
    
    for i in range(len(drivers_list)):
        color_hex = hex_palette[i % len(hex_palette)]
        # Convert hex to RGBA with 0.4 alpha for the fill
        rgba = mcolors.to_rgba(color_hex, alpha=0.4)
        
        bar_fills.append(rgba)
        bar_edges.append(color_hex)

    # 4. Generate the Plot
    plt.figure(figsize=(10, 6))

    # Create horizontal bar chart
    bars = plt.barh(
        drivers_list, 
        total_counts, 
        color=bar_fills, 
        edgecolor=bar_edges,
        linewidth=2 # Thicker border for definition
    )

    # 5. Styling
    plt.xlabel('Total Frequency', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    plt.ylabel('Interaction Drivers', fontsize=12, fontweight='bold', color=MIDNIGHT_VIOLET)
    plt.title('Total Frequency of Interaction Drivers', fontsize=14, fontweight='bold', color=MIDNIGHT_VIOLET)

    # Configure Grid
    plt.grid(axis='x', linestyle=':', alpha=0.6, color='#651FFF')
    
    # Configure Axes Tick Colors
    ax = plt.gca()
    ax.tick_params(axis='x', colors=MIDNIGHT_VIOLET)
    ax.tick_params(axis='y', colors=MIDNIGHT_VIOLET)

    # Force x-axis to show integer ticks
    if total_counts:
        # Add a little buffer to the max limit so labels fit
        max_val = max(total_counts)
        plt.xticks(range(int(max_val) + 2))

    # Add count labels at the end of the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Use the solid edge color for the text label
        label_color = bar_edges[i]
        
        plt.text(
            width + 0.1,       # x-position (slightly to the right of the bar)
            bar.get_y() + bar.get_height()/2, # y-position (center of bar)
            f'{int(width)}', 
            va='center', 
            fontweight='bold',
            color=label_color
        )

    plt.tight_layout()
    plt.show()

# --- Example Execution ---
if __name__ == "__main__":
    file_path = 'Corpus\corpus.json'
    target_drivers = [
        'customer_dissent', 
        'service_degradation', 
        'satisfaction_expression', 
        'churn_or_cancellation_behavior',
        'customer_appreciation'
    ]
    target_ids = ['c94a0d68-e86a-46b2-8c16-805563f03462']

    plot_driver_occurrences(file_path, target_drivers, target_ids)